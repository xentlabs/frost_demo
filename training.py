from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList


PLAYER_PROMPT = (
    "I will give you a text. Output a title that is maximally informative about the text. "
    "The title should consist of at most {title_len} tokens. "
    "The title should not contain any tokens that are also in the text. "
    "Output only that title and nothing else.\n"
    "The text:\n{target}"
)
PRE_SANDWICH = "Title:"
POST_SANDWICH = "\n\n"

DATASETS = {
    "cosmopedia": {
        "hf_dataset": "HuggingFaceTB/cosmopedia",
        "hf_subset": "stories",
        "text_field": "text",
    },
    "fineweb-edu": {
        "hf_dataset": "HuggingFaceFW/fineweb-edu",
        "hf_subset": "sample-10BT",
        "text_field": "text",
    },
}

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass
class FrostTrainingResults:
    model: Any = field(repr=False)
    tokenizer: Any = field(repr=False)
    validation_texts: list[str]
    validation_steps: list[int]
    validation_outputs: dict[int, list[list[str]]]
    validation_scores: dict[int, list[list[float]]]
    config: dict[str, Any]


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _dtype(device):
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def _load_causal_lm(model_name, device):
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=_dtype(device),
        low_cpu_mem_usage=True,
    ).to(device)


def _scoring_bos_id(model, tokenizer):
    if model.config.bos_token_id is not None:
        return model.config.bos_token_id
    return tokenizer.eos_token_id


def _model_device(model):
    return next(model.parameters()).device


def _vocab_size(model):
    return model.get_input_embeddings().weight.shape[0]


def _tokenize_context(tokenizer, device, text):
    return tokenizer([text], return_tensors="pt", add_special_tokens=False).input_ids.to(device)


def _build_prompt_ids(tokenizer, target_ids, title_len, device):
    target_marker = "<<TARGET>>"
    content = PLAYER_PROMPT.format(title_len=title_len, target=target_marker)
    messages = [{"role": "user", "content": content}]
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    before, after = prompt.split(target_marker)
    return torch.cat(
        [
            _tokenize_context(tokenizer, device, before),
            target_ids.to(device),
            _tokenize_context(tokenizer, device, after),
        ],
        dim=1,
    )


def _iter_chunks(items, chunk_size):
    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size]


def _load_streamed_texts(dataset, tokenizer, text_len, num_validation_texts, num_train_texts):
    config = DATASETS[dataset]
    ds = load_dataset(
        config["hf_dataset"],
        config["hf_subset"],
        split="train",
        streaming=True,
    )
    val_targets = []
    val_texts = []
    train_targets = []
    total_needed = num_validation_texts + num_train_texts
    seen = 0

    print(f"Downloading {total_needed} texts from {config['hf_dataset']}/{config['hf_subset']}...")
    for sample in ds:
        text = sample[config["text_field"]].strip()
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) < text_len:
            continue

        target_ids = torch.tensor(token_ids[:text_len], dtype=torch.long).unsqueeze(0)
        if len(val_targets) < num_validation_texts:
            val_targets.append(target_ids)
            val_texts.append(tokenizer.decode(token_ids[:text_len]))
        else:
            train_targets.append(target_ids)

        seen += 1
        if seen >= total_needed:
            break

    if len(val_targets) < num_validation_texts or len(train_targets) < num_train_texts:
        raise RuntimeError(
            f"Only found {len(val_targets)} validation and {len(train_targets)} training texts of the required length."
        )
    return val_targets, val_texts, train_targets


def _forbidden_token_mask(tokenizer, targets, num_tokens, device):
    always_banned = list(tokenizer.added_tokens_decoder)
    always_banned.extend(tokenizer.encode("\n", add_special_tokens=False))
    always_banned = torch.tensor(always_banned, dtype=torch.long, device=device)

    mask = torch.zeros(len(targets), num_tokens, dtype=torch.bool, device=device)
    for i, target in enumerate(targets):
        banned = target.to(device=device, dtype=torch.long).flatten()
        banned = torch.cat([banned, always_banned])
        mask[i, torch.unique(banned)] = True
    return mask


def _generation_pad_token_id(tokenizer):
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    return tokenizer.eos_token_id


class _ForbiddenTokenMaskLogitsProcessor(LogitsProcessor):
    def __init__(self, forbidden_mask, num_return_sequences):
        self.forbidden_mask = forbidden_mask.repeat_interleave(num_return_sequences, dim=0)

    def __call__(self, input_ids, scores):
        return scores.masked_fill(self.forbidden_mask, float("-inf"))


@torch.no_grad()
def _sample_titles(model, prompt_ids, title_len, num_samples, forbidden_mask, pad_token_id):
    _, prompt_len = prompt_ids.shape
    logits_processor = LogitsProcessorList([
        _ForbiddenTokenMaskLogitsProcessor(forbidden_mask, num_samples)
    ])
    out = model.generate(
        prompt_ids,
        attention_mask=torch.ones_like(prompt_ids),
        max_new_tokens=title_len,
        min_new_tokens=title_len,
        do_sample=True,
        temperature=1.0,
        num_return_sequences=num_samples,
        logits_processor=logits_processor,
        pad_token_id=pad_token_id,
    )
    return out[:, prompt_len:prompt_len + title_len].view(
        prompt_ids.shape[0], num_samples, title_len)


def _title_logits(model, prompt_ids, titles):
    _, title_len = titles.shape
    prompt_len = prompt_ids.shape[1]
    seq = torch.cat([prompt_ids, titles], dim=1)
    return model(seq, use_cache=False).logits[:, prompt_len - 1:prompt_len - 1 + title_len, :].float()


def _mask_forbidden_logits(logits, forbidden_mask):
    return logits.masked_fill(forbidden_mask, float("-inf"))


def _policy_logprobs(model, prompt_ids, titles, forbidden_mask):
    logits = _mask_forbidden_logits(
        _title_logits(model, prompt_ids, titles),
        forbidden_mask.unsqueeze(1),
    )
    full_lp = F.log_softmax(logits, dim=-1)
    token_lp = full_lp.gather(2, titles.unsqueeze(-1)).squeeze(-1)
    return token_lp, full_lp


@torch.no_grad()
def _ref_logprobs(model, prompt_ids, titles, forbidden_mask):
    logits = _mask_forbidden_logits(
        _title_logits(model, prompt_ids, titles),
        forbidden_mask.unsqueeze(1),
    )
    return F.log_softmax(logits, dim=-1)


def _token_ce(logits, labels):
    batch, seq_len = labels.shape
    return F.cross_entropy(
        logits.reshape(batch * seq_len, -1),
        labels.reshape(-1),
        reduction="none",
    ).float().view(batch, seq_len)


def _score_titles(model, bos_id, pre_ids, post_ids, titles, target_ids):
    batch, title_len = titles.shape
    device = titles.device
    pre_len = pre_ids.shape[1]
    post_len = post_ids.shape[1]
    bos = torch.full((batch, 1), bos_id, dtype=torch.long, device=device)
    pre = pre_ids.to(device).expand(batch, -1)
    post = post_ids.to(device).expand(batch, -1)
    target = target_ids.to(device).expand(batch, -1)
    seq = torch.cat([bos, pre, titles, post, target], dim=1)

    logits = model(seq, use_cache=False).logits
    ce_title = _token_ce(logits[:, pre_len:pre_len + title_len, :], titles).sum(dim=1)
    text_start = pre_len + title_len + post_len
    ce_text = _token_ce(logits[:, text_start:text_start + target.shape[1], :], target).sum(dim=1)
    return -(ce_title + ce_text)


@torch.no_grad()
def _score_unique_titles(model, bos_id, pre_ids, post_ids, titles, target_ids):
    unique, inverse = torch.unique(titles, dim=0, return_inverse=True)
    scores = _score_titles(model, bos_id, pre_ids, post_ids, unique, target_ids)
    return scores[inverse]


def _taylor_score_approx(model, bos_id, pre_ids, post_ids, titles, target_ids):
    batch, title_len = titles.shape
    device = titles.device
    embed = model.get_input_embeddings()
    weight = embed.weight
    pre_len = pre_ids.shape[1]
    post_len = post_ids.shape[1]

    bos = torch.full((batch, 1), bos_id, dtype=torch.long, device=device)
    pre = pre_ids.to(device).expand(batch, -1)
    post = post_ids.to(device).expand(batch, -1)
    target = target_ids.to(device).expand(batch, -1)

    title_embeds = embed(titles).detach().requires_grad_(True)
    seq_embeds = torch.cat([
        embed(bos).detach(),
        embed(pre).detach(),
        title_embeds,
        embed(post).detach(),
        embed(target).detach(),
    ], dim=1)

    logits = model(inputs_embeds=seq_embeds, use_cache=False).logits
    text_start = pre_len + title_len + post_len
    loss = _token_ce(logits[:, text_start:text_start + target.shape[1], :], target).sum(dim=1)
    loss = loss + _token_ce(logits[:, pre_len:pre_len + title_len, :], titles).sum(dim=1)

    grad = torch.autograd.grad(loss.sum(), title_embeds)[0]
    score_center = -loss.detach()

    base_dots = (grad * title_embeds.detach()).sum(dim=-1)
    all_dots = torch.matmul(grad, weight.T)
    approx = score_center.view(batch, 1, 1) - (all_dots - base_dots.unsqueeze(-1))

    title_lp = F.log_softmax(logits[:, pre_len:pre_len + title_len, :].detach().float(), dim=-1)
    lp_at_y = title_lp.gather(2, titles.unsqueeze(-1))
    approx = approx + title_lp - lp_at_y
    return approx.detach()


def _discover_mutations(player_model, judge_model, judge_bos_id, pre_ids, post_ids,
                        target_ids, prompt_ids, titles, forbidden_mask,
                        shared_player_judge, probability_gate, num_frost_samples,
                        batch_taylor_approximations):
    num_rollouts, title_len = titles.shape
    device = titles.device

    player_model.enable_adapter_layers()
    logits = _mask_forbidden_logits(
        _title_logits(player_model, prompt_ids.expand(num_rollouts, -1), titles),
        forbidden_mask.view(1, 1, -1),
    )
    probs = F.softmax(logits, dim=-1).detach()
    vocab = probs.shape[-1]

    if shared_player_judge:
        player_model.disable_adapter_layers()
    if batch_taylor_approximations:
        approx = _taylor_score_approx(
            judge_model, judge_bos_id, pre_ids, post_ids, titles, target_ids)
    else:
        approx = torch.cat([
            _taylor_score_approx(judge_model, judge_bos_id, pre_ids, post_ids, title, target_ids)
            for title in titles.split(1, dim=0)
        ], dim=0)
    if shared_player_judge:
        player_model.enable_adapter_layers()

    noop = torch.zeros(num_rollouts, title_len, vocab, dtype=torch.bool, device=device)
    noop.scatter_(2, titles.unsqueeze(-1), True)
    illegal = noop | forbidden_mask.view(1, 1, vocab)
    rank_key = approx.masked_fill(illegal | (probs <= probability_gate), float("-inf"))

    num_valid = int(torch.isfinite(rank_key).sum().item())
    num_top = min(num_frost_samples, num_valid)
    if num_top == 0:
        empty = torch.empty(0, title_len, dtype=titles.dtype, device=device)
        return empty, torch.empty(0, device=device), []

    top_idx = rank_key.reshape(-1).topk(num_top).indices
    selected_token = top_idx % vocab
    rem = top_idx // vocab
    selected_pos = rem % title_len
    selected_rollout = rem // title_len

    mutated = titles[selected_rollout].clone()
    mutated[torch.arange(num_top, device=device), selected_pos] = selected_token.to(titles.dtype)

    with torch.no_grad():
        if shared_player_judge:
            player_model.disable_adapter_layers()
        scores = _score_unique_titles(
            judge_model, judge_bos_id, pre_ids, post_ids, mutated, target_ids).detach()
        if shared_player_judge:
            player_model.enable_adapter_layers()

    return mutated, scores, selected_rollout.tolist()


def _build_replacement_group(titles, sampled_scores, mutated, mutated_scores, origins):
    candidates = titles.clone()
    scores = sampled_scores.clone()
    best: dict[int, tuple[float, int]] = {}
    for idx in range(mutated.shape[0]):
        rollout_idx = int(origins[idx])
        score = float(mutated_scores[idx].item())
        if score <= float(sampled_scores[rollout_idx].item()):
            continue
        if rollout_idx not in best or score > best[rollout_idx][0]:
            best[rollout_idx] = (score, idx)
    for rollout_idx, (_, idx) in best.items():
        candidates[rollout_idx] = mutated[idx]
        scores[rollout_idx] = mutated_scores[idx]
    return candidates, scores


def _reinforce_step_batched(model, prompt_ids, titles, scores, kl_regularizer,
                            forbidden_mask):
    batch, rollouts, title_len = titles.shape
    prompt_len = prompt_ids.shape[1]
    total = batch * rollouts

    prompts_flat = prompt_ids.unsqueeze(1).expand(batch, rollouts, prompt_len).reshape(total, prompt_len)
    titles_flat = titles.reshape(total, title_len)
    forbidden_flat = forbidden_mask.repeat_interleave(rollouts, dim=0)

    baseline = scores.mean(dim=1).detach()
    advantages = (scores - baseline.unsqueeze(1)).detach().reshape(total)

    token_lp, full_lp = _policy_logprobs(model, prompts_flat, titles_flat, forbidden_flat)
    model.disable_adapter_layers()
    ref_lp = _ref_logprobs(model, prompts_flat, titles_flat, forbidden_flat)
    model.enable_adapter_layers()

    forbidden_pos = forbidden_flat.unsqueeze(1)
    policy_p = full_lp.exp().masked_fill(forbidden_pos, 0.0)
    log_diff = (full_lp - ref_lp).masked_fill(forbidden_pos, 0.0)
    kl = (policy_p * log_diff).sum(dim=-1)
    kl_sum = kl.sum(dim=1)
    log_pi = token_lp.sum(dim=1)
    loss = (-advantages * log_pi + kl_regularizer * kl_sum).sum() / rollouts
    loss.backward()

    return kl_sum.view(batch, rollouts).mean().item()


def _setup_models(player_model_name, judge_model_name, lora_rank, device):
    print(f"Loading player model: {player_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(player_model_name, trust_remote_code=True)
    player_base = _load_causal_lm(player_model_name, device)
    player = get_peft_model(player_base, LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    ))
    player.train()

    shared_player_judge = player_model_name == judge_model_name
    if shared_player_judge:
        print("Using player model as judge with adapters disabled for scoring.")
        judge = player
        judge_tokenizer = tokenizer
    else:
        print(f"Loading judge model: {judge_model_name}")
        judge = _load_causal_lm(judge_model_name, device)
        judge.eval()
        judge.requires_grad_(False)
        judge_tokenizer = AutoTokenizer.from_pretrained(judge_model_name, trust_remote_code=True)

    if _vocab_size(player) != _vocab_size(judge):
        raise ValueError("player and judge vocabularies must have the same size")
    judge_bos_id = _scoring_bos_id(judge, judge_tokenizer)
    if judge_bos_id is None:
        raise ValueError("judge must define either bos_token_id or eos_token_id")
    return player, tokenizer, judge, judge_tokenizer, judge_bos_id, shared_player_judge


def _score_batch(judge_model, judge_bos_id, pre_ids, post_ids, titles, targets,
                 shared_player_judge, player_model):
    if shared_player_judge:
        player_model.disable_adapter_layers()
    with torch.no_grad():
        per_target = [
            _score_unique_titles(judge_model, judge_bos_id, pre_ids, post_ids, titles[i], target)
            for i, target in enumerate(targets)
        ]
    if shared_player_judge:
        player_model.enable_adapter_layers()
    return torch.stack(per_target, dim=0)


def _train_step(player_model, player_tokenizer, judge_model, judge_bos_id,
                pre_ids, post_ids, shared_player_judge, targets, optimizer,
                title_len, num_rollouts_per_text, num_frost_samples_per_text,
                micro_batch_size, kl_regularizer, probability_gate,
                batch_taylor_approximations):
    player_model.enable_adapter_layers()
    player_model.zero_grad(set_to_none=True)
    total_kl = 0.0
    total_texts = 0
    device = _model_device(player_model)

    for batch_targets in _iter_chunks(targets, micro_batch_size):
        prompt_ids = torch.cat([
            _build_prompt_ids(player_tokenizer, target, title_len, device)
            for target in batch_targets
        ], dim=0)
        forbidden_mask = _forbidden_token_mask(
            player_tokenizer, batch_targets, _vocab_size(player_model), device)
        titles = _sample_titles(
            player_model,
            prompt_ids,
            title_len,
            num_rollouts_per_text,
            forbidden_mask,
            _generation_pad_token_id(player_tokenizer),
        )

        sampled_scores = _score_batch(
            judge_model, judge_bos_id, pre_ids, post_ids, titles, batch_targets,
            shared_player_judge, player_model)

        candidate_titles = []
        candidate_scores = []
        for i, target in enumerate(batch_targets):
            mutated, mutated_scores, origins = _discover_mutations(
                player_model,
                judge_model,
                judge_bos_id,
                pre_ids,
                post_ids,
                target,
                prompt_ids[i:i + 1],
                titles[i],
                forbidden_mask[i],
                shared_player_judge,
                probability_gate,
                num_frost_samples_per_text,
                batch_taylor_approximations,
            )
            candidates, scores = _build_replacement_group(
                titles[i], sampled_scores[i], mutated, mutated_scores, origins)
            candidate_titles.append(candidates)
            candidate_scores.append(scores)

        candidate_titles = torch.stack(candidate_titles, dim=0)
        candidate_scores = torch.stack(candidate_scores, dim=0)
        mean_kl = _reinforce_step_batched(
            player_model,
            prompt_ids,
            candidate_titles,
            candidate_scores,
            kl_regularizer,
            forbidden_mask,
        )
        total_kl += mean_kl * len(batch_targets)
        total_texts += len(batch_targets)

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return total_kl / total_texts


def _validation_summary(scores_by_text):
    flat_scores = [score for row in scores_by_text for score in row]
    mean_score = sum(flat_scores) / len(flat_scores)
    best_score = sum(max(row) for row in scores_by_text) / len(scores_by_text)
    return mean_score, best_score


def _validate(player_model, player_tokenizer, judge_model, judge_bos_id,
              pre_ids, post_ids, shared_player_judge, val_targets, title_len,
              num_validation_rollouts, validation_batch_size):
    player_model.eval()
    device = _model_device(player_model)
    all_outputs = []
    all_scores = []

    for batch_targets in _iter_chunks(val_targets, validation_batch_size):
        prompt_ids = torch.cat([
            _build_prompt_ids(player_tokenizer, target, title_len, device)
            for target in batch_targets
        ], dim=0)
        forbidden_mask = _forbidden_token_mask(
            player_tokenizer, batch_targets, _vocab_size(player_model), device)
        player_model.enable_adapter_layers()
        titles = _sample_titles(
            player_model,
            prompt_ids,
            title_len,
            num_validation_rollouts,
            forbidden_mask,
            _generation_pad_token_id(player_tokenizer),
        )
        scores = _score_batch(
            judge_model, judge_bos_id, pre_ids, post_ids, titles, batch_targets,
            shared_player_judge, player_model)

        for i in range(len(batch_targets)):
            all_outputs.append([
                player_tokenizer.decode(titles[i, j].tolist())
                for j in range(num_validation_rollouts)
            ])
            all_scores.append([
                float(scores[i, j].item())
                for j in range(num_validation_rollouts)
            ])

    player_model.train()
    return all_outputs, all_scores


def train_with_frost(
    dataset="cosmopedia",
    text_len=128,
    title_len=10,
    player_model="Qwen/Qwen3.5-2B",
    judge_model="Qwen/Qwen3.5-2B",
    num_steps=20,
    num_texts_per_step=4,
    num_rollouts_per_text=4,
    num_frost_samples_per_text=4,
    micro_batch_size=4,
    lr=1e-5,
    kl_regularizer=0.1,
    probability_gate=1e-4,
    lora_rank=64,
    num_validation_texts=16,
    num_validation_rollouts=8,
    validation_steps=10,
    batch_taylor_approximations=False,
):
    device = _device()
    print(f"Using device: {device}")
    model, tokenizer, judge, judge_tokenizer, judge_bos_id, shared_player_judge = _setup_models(
        player_model, judge_model, lora_rank, device)

    val_targets, val_texts, train_targets = _load_streamed_texts(
        dataset,
        tokenizer,
        text_len,
        num_validation_texts,
        num_steps * num_texts_per_step,
    )

    pre_ids = _tokenize_context(judge_tokenizer, _model_device(judge), PRE_SANDWICH)
    post_ids = _tokenize_context(judge_tokenizer, _model_device(judge), POST_SANDWICH)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=lr,
    )

    results = FrostTrainingResults(
        model=model,
        tokenizer=tokenizer,
        validation_texts=val_texts,
        validation_steps=[],
        validation_outputs={},
        validation_scores={},
        config={
            "dataset": dataset,
            "text_len": text_len,
            "title_len": title_len,
            "player_model": player_model,
            "judge_model": judge_model,
            "num_steps": num_steps,
            "num_texts_per_step": num_texts_per_step,
            "num_rollouts_per_text": num_rollouts_per_text,
            "num_frost_samples_per_text": num_frost_samples_per_text,
            "micro_batch_size": micro_batch_size,
            "lr": lr,
            "kl_regularizer": kl_regularizer,
            "probability_gate": probability_gate,
            "lora_rank": lora_rank,
            "num_validation_texts": num_validation_texts,
            "num_validation_rollouts": num_validation_rollouts,
            "validation_steps": validation_steps,
            "batch_taylor_approximations": batch_taylor_approximations,
        },
    )

    validation_batch_size = 4 * micro_batch_size

    def run_validation(step):
        outputs, scores = _validate(
            model,
            tokenizer,
            judge,
            judge_bos_id,
            pre_ids,
            post_ids,
            shared_player_judge,
            val_targets,
            title_len,
            num_validation_rollouts,
            validation_batch_size,
        )
        results.validation_steps.append(step)
        results.validation_outputs[step] = outputs
        results.validation_scores[step] = scores
        mean_score, best_score = _validation_summary(scores)
        print(
            f"step {step:4d} | mean validation score: {mean_score:.2f} | "
            f"best-of-{num_validation_rollouts}: {best_score:.2f}"
        )

    run_validation(0)
    for step in range(1, num_steps + 1):
        start = (step - 1) * num_texts_per_step
        end = start + num_texts_per_step
        _train_step(
            model,
            tokenizer,
            judge,
            judge_bos_id,
            pre_ids,
            post_ids,
            shared_player_judge,
            train_targets[start:end],
            optimizer,
            title_len,
            num_rollouts_per_text,
            num_frost_samples_per_text,
            micro_batch_size,
            kl_regularizer,
            probability_gate,
            batch_taylor_approximations,
        )
        if validation_steps > 0 and step % validation_steps == 0:
            run_validation(step)

    if results.validation_steps[-1] != num_steps:
        run_validation(num_steps)

    return results
