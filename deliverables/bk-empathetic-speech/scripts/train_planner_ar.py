"""Bounded AR feasibility adaptation and evaluation. / 제한된 AR 타당성 적응 학습과 평가."""

import argparse
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils.rnn import pad_sequence

from dataset.quality_speech_dataset import QualitySpeechDataset, collate_quality, person_a_only
from model.full_speech.experimental_ar_units import AR_ARCHITECTURE, ar_system_from_payload, load_ar_checkpoint
from model.full_speech.tensor_ops import counts, mask_from_lengths
from prepare_quality import atomic_json
from scripts.check_unit_vocabulary_audio import file_hash, module_hashes, tensor_hash
from train_full import atomic_save, move_batch
from utils.distributed_training import DistributedRuntime, LossForward


def selected_records(manifest_path, selection_path, split):
    manifest = json.loads(manifest_path.read_text())
    selection = json.loads(selection_path.read_text())
    if selection.get('manifest_sha256') != file_hash(manifest_path):
        raise ValueError('Manifest and selection hashes differ')
    wanted = selection.get(split, [])
    if not wanted or len(wanted) != len(set(wanted)) or set(selection.get('train', [])) & set(selection.get('val', [])):
        raise ValueError('Selections must be nonempty, unique and split-disjoint')
    mapping = {row['path']: row for row in manifest['records']}
    if len(mapping) != len(manifest['records']) or any(path not in mapping or mapping[path]['split'] != split for path in wanted):
        raise ValueError('Selected paths missing or assigned to a different split')
    conversation_sets = {}
    for group in ('train', 'val'):
        paths = selection.get(group, [])
        if any(path not in mapping or mapping[path]['split'] != group for path in paths):
            raise ValueError('Selection changes original split membership')
        conversation_sets[group] = {mapping[path]['conversation_id'] for path in paths}
    if conversation_sets['train'] & conversation_sets['val']:
        raise ValueError('Training and validation conversations overlap')
    rows = [mapping[path] for path in wanted]
    if len({row['conversation_id'] for row in rows}) != len(rows):
        raise ValueError('Repeated conversations in selection')
    return rows


class SelectedSamples:
    """Load conditional caches lazily to bound host memory. / 조건 캐시를 지연 로딩하여 메모리를 제한합니다."""
    def __init__(self, manifest, selection, model, split, phase, conditional_cache='lazy'):
        began = time.monotonic()
        self.records = selected_records(manifest, selection, split)
        self.phase = phase
        self.cached = None
        if phase == 'prior':
            self.units = []
            with torch.inference_mode():
                for row in self.records:
                    with np.load(manifest.parent / row['path'], allow_pickle=False) as cache:
                        semantic = torch.from_numpy(cache['semantic'].astype(np.float32)).to(model.semantic_mean.device)
                    if semantic.ndim != 2 or semantic.shape[1] != 768 or not len(semantic) or not torch.isfinite(semantic).all():
                        raise ValueError('Invalid B semantic cache')
                    ids = model.semantic_planner.codebook.encode((semantic - model.semantic_mean) / model.semantic_std)
                    self.units.append(ids.cpu())
        else:
            self.data = QualitySpeechDataset(manifest, model.config, split)
            mapping = {row['path']: i for i, row in enumerate(self.data.records)}
            self.indices = [mapping[row['path']] for row in self.records]
            if conditional_cache == 'inputs_units':
                self.cached = []
                with torch.inference_mode():
                    for index in self.indices:
                        sample = self.data[index]
                        semantic = sample['semantic'].to(model.semantic_mean.device)
                        ids = model.semantic_planner.codebook.encode((semantic - model.semantic_mean) / model.semantic_std)
                        # Keep only A inputs and B unit labels in RAM. / RAM에는 A 입력과 B 단위 라벨만 보존합니다.
                        self.cached.append({**{key: sample[key] for key in
                            ('mel', 'dmm', 'au', 'speech_a', 'style_id', 'speaker_id')}, 'unit_ids': ids.cpu()})
        self.profile = {'phase': phase, 'examples': len(self), 'conditional_cache': conditional_cache,
                        'preparation_seconds': time.monotonic() - began,
                        'cached_tensor_bytes': (sum(value.numel() * value.element_size() for row in self.cached
                            for value in row.values()) if self.cached is not None else
                            sum(value.numel() * value.element_size() for value in self.units) if phase == 'prior' else 0)}

    def __len__(self):
        return len(self.records)

    def batch(self, indices, device):
        if self.phase == 'prior':
            rows = [self.units[index] for index in indices]
            return {'unit_ids': pad_sequence(rows, batch_first=True).to(device),
                    'unit_len': torch.tensor([len(row) for row in rows], device=device)}
        if self.cached is not None:
            rows = [self.cached[index] for index in indices]
            batch = {}
            for key in ('mel', 'dmm', 'au', 'speech_a', 'unit_ids'):
                batch[key] = pad_sequence([row[key] for row in rows], batch_first=True).to(device)
                length_key = 'unit_len' if key == 'unit_ids' else key + '_len'
                batch[length_key] = torch.tensor([len(row[key]) for row in rows], device=device)
            for key in ('style_id', 'speaker_id'):
                batch[key] = torch.stack([row[key] for row in rows]).to(device)
            return batch
        return move_batch(collate_quality([self.data[self.indices[index]] for index in indices]), device)


def frozen_hashes(model):
    result = {key: value for key, value in module_hashes(model).items() if key != 'semantic_planner'}
    result['fixed_codebook'] = tensor_hash(model.semantic_planner.codebook.centers)
    return result


def verify_feasibility_shape(model):
    if (len(model.semantic_planner.codebook.centers), model.config.hidden_dim, model.config.planner_layers) != (1024, 256, 4):
        raise ValueError('Real AR feasibility runs require unchanged 1024 units, width 256 and four planner blocks')


def frozen_audit(model, expected):
    current = frozen_hashes(model)
    changed = sorted(key for key in expected if current.get(key) != expected[key])
    nonplanner = [name for name, parameter in model.named_parameters()
                  if not name.startswith('semantic_planner.') and parameter.grad is not None]
    result = {'passed': not changed and not nonplanner, 'frozen_changed': changed,
              'nonplanner_gradients': nonplanner, 'frozen_hashes': current}
    if not result['passed']:
        raise RuntimeError('AR feasibility run changed a frozen component')
    return result


@torch.inference_mode()
def evaluate_teacher(model, data, runtime, batch_size):
    model.eval()
    totals = torch.zeros(5, dtype=torch.float64, device=runtime.device)
    local = list(range(runtime.rank, len(data), runtime.world_size))
    for offset in range(0, len(local), batch_size):
        indices = local[offset:offset + batch_size]
        batch = data.batch(indices, runtime.device)
        values = model.teacher_outputs(batch)
        if not torch.isfinite(values['loss']):
            raise RuntimeError('Non-finite teacher-forced validation')
        totals += torch.tensor([float(values['loss']) * len(indices), float(values['ce_sum']),
            int(values['correct']), int(values['frames']), len(indices)], dtype=totals.dtype, device=totals.device)
    if runtime.distributed:
        torch.distributed.all_reduce(totals)
    if totals[4] < 1 or totals[3] < 1:
        raise ValueError('Empty held-out evaluation')
    return {'example_mean_next_unit_ce': float(totals[0] / totals[4]),
            'frame_next_unit_ce': float(totals[1] / totals[3]),
            'teacher_forced_unit_accuracy': float(totals[2] / totals[3]),
            'frames': int(totals[3]), 'examples': int(totals[4]),
            'uses_correct_B_prefix': True, 'oracle_B_length': True, 'normal_inference': False}


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, unit in enumerate(left, 1):
        current = [i]
        for j, target in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (unit != target)))
        previous = current
    return previous[-1]


@torch.inference_mode()
def evaluate_free(model, data, device, count):
    model.eval()
    examples = []
    for index in range(min(count, len(data))):
        batch = data.batch([index], device)
        target, valid = model.unit_targets(batch)
        if model.ar_phase == 'prior':
            mask = valid
            duration_source = 'oracle_B_length_for_unconditional_prior_diagnostic'
            conditions = {'null_prior': model.planner_conditions(batch, 1)}
            donor = None
        else:
            if len(data) < 2:
                raise ValueError('Shuffled-A free evaluation needs a second conversation')
            inputs = person_a_only(batch)
            encoded = model.encode_batch(inputs)
            style, _ = model.embeddings(inputs['style_id'], inputs['speaker_id'], 1)
            duration, _ = model.length_predictor(encoded['context'], encoded['affect'], encoded['context_mask'], style)
            mask = mask_from_lengths(counts(duration, model.config.semantic_hz))
            duration_source = 'A_predicted_duration'
            donor = (index + 1) % len(data)
            donor_inputs = person_a_only(data.batch([donor], device))
            donor_inputs.update(style_id=inputs['style_id'], speaker_id=inputs['speaker_id'])
            swapped = model.encode_batch(donor_inputs)
            conditions = {'correct_a': {**model.planner_inputs(encoded), 'style': style},
                          'shuffled_a': {**model.planner_inputs(swapped), 'style': style}}
        expected = target[valid].cpu().tolist()
        entry = {'path': data.records[index]['path'], 'conversation_id': data.records[index]['conversation_id'],
            'target_unit_ids': expected, 'predicted_frames': int(mask.sum()), 'reference_frames': len(expected),
            'duration_source': duration_source, 'normal_inference': False,
            'A_only_generation': model.ar_phase == 'conditional',
            'uses_B_unit_inputs': False, 'conditions': {}, 'style_and_speaker_held_fixed': True,
            'shuffled_path': data.records[donor]['path'] if donor is not None else None}
        for name, memory in conditions.items():
            began = time.monotonic()
            ids = model.semantic_planner.sample_ids(mask, **memory)[mask].cpu().tolist()
            overlap = min(len(ids), len(expected))
            entry['conditions'][name] = {'generated_unit_ids': ids,
                'normal_inference': name == 'correct_a', 'uses_B_unit_inputs': False,
                'unit_edit_distance_per_reference_unit': edit_distance(ids, expected) / len(expected),
                'prefix_aligned_accuracy': sum(a == b for a, b in zip(ids, expected)) / max(1, overlap),
                'adjacent_repeat_fraction': sum(a == b for a, b in zip(ids, ids[1:])) / max(1, len(ids) - 1),
                'generated_distinct_units': len(set(ids)), 'generation_seconds': time.monotonic() - began}
        if donor is not None:
            a, b = (entry['conditions'][name]['generated_unit_ids'] for name in ('correct_a', 'shuffled_a'))
            entry['changed_unit_fraction_under_shuffled_a'] = sum(x != y for x, y in zip(a, b)) / len(a)
        examples.append(entry)
    return {'count': len(examples), 'examples': examples,
        'summary': {name: {key: sum(row['conditions'][name][key] for row in examples) / len(examples) for key in
            ('unit_edit_distance_per_reference_unit', 'prefix_aligned_accuracy', 'adjacent_repeat_fraction', 'generation_seconds')}
            for name in examples[0]['conditions']},
        'A_encoder_inputs': 'person_a_only', 'duration_locked_across_controls': True,
        'limitations': ['Unit match to one B response does not measure relevance; duration differences affect alignment.',
                        'Shuffled-A responses use original A-predicted duration, not donor duration; sensitivity is not correctness.',
                        'Uncached AR latency is reported; this is not a matched-compute architecture comparison.']}


def train(args, runtime):
    if args.output.joinpath('recipe.json').exists() and not args.resume:
        raise ValueError('Use a new training output directory')
    if runtime.distributed:
        torch.distributed.barrier()
    torch.manual_seed(args.seed)
    source = args.resume or args.initialize
    payload = torch.load(source, map_location='cpu', weights_only=True)
    model = ar_system_from_payload(payload, allow_masked_initialization=not bool(args.resume)).to(runtime.device)
    verify_feasibility_shape(model)
    model.configure_ar_training(args.phase)
    previous_recipe = payload.get('metadata', {}).get('ar_recipe', {})
    recipe = {'phase': args.phase, 'steps': args.steps, 'batch_size_per_gpu': args.batch_size,
        'world_size': runtime.world_size, 'lr': args.lr, 'seed': args.seed,
        'manifest_sha256': file_hash(args.manifest), 'selection_sha256': file_hash(args.selection),
        'source_checkpoint_sha256': previous_recipe.get('source_checkpoint_sha256') if args.resume else file_hash(source),
        'initialization_architecture': previous_recipe.get('initialization_architecture') if args.resume else payload['architecture'],
        'loss': 'equal-example causal next-unit CE', 'duration_and_acoustics_frozen': True,
        'evaluation_every': args.evaluate_every, 'precision': 'float32', 'feasibility_only': True,
        'conditional_cache': args.conditional_cache,
        'comparison_limit': 'Extra prior adaptation and serial generation; not an equal-compute architecture winner.'}
    repo = Path(__file__).resolve().parents[1]
    recipe['source_sha256'] = {name: file_hash(repo / name) for name in (
        'scripts/train_planner_ar.py', 'model/full_speech/experimental_ar_units.py', 'model/full_speech/dit.py',
        'model/full_speech/units.py', 'model/full_speech/quality.py', 'model/full_speech/tensor_ops.py',
        'dataset/quality_speech_dataset.py')}
    if args.resume and previous_recipe != recipe:
        raise ValueError('AR resume recipe changed')
    frozen = frozen_hashes(model)
    if args.resume and payload.get('ar_frozen_hashes') != frozen:
        raise ValueError('Frozen source hash changed before AR resume')
    training = SelectedSamples(args.manifest, args.selection, model, 'train', args.phase, args.conditional_cache)
    validation = SelectedSamples(args.manifest, args.selection, model, 'val', args.phase, args.conditional_cache)
    global_batch = args.batch_size * runtime.world_size
    if len(training) < global_batch or len(training) % global_batch or len(validation) < runtime.world_size:
        raise ValueError('Need full unique global batches and validation examples on every rank')
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=.01)

    def schedule(step):
        warmup = min(50, max(1, args.steps // 10))
        if step < warmup:
            return .1 + .9 * step / warmup
        return .1 + .9 * .5 * (1 + math.cos(math.pi * min(1., (step - warmup) / max(1, args.steps - warmup))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    generator = torch.Generator().manual_seed(args.seed + runtime.rank)
    start = 0
    if args.resume:
        optimizer.load_state_dict(payload['optimizer'])
        scheduler.load_state_dict(payload['scheduler'])
        start = payload['ar_step']
        rng = payload['rank_rng_states'][runtime.rank]
        torch.set_rng_state(rng['torch_rng'])
        generator.set_state(rng['loader_rng'])
        if runtime.device.type == 'cuda':
            torch.cuda.set_rng_state(rng['cuda_rng'], runtime.device)
    else:
        torch.manual_seed(args.seed + runtime.rank)
    limit = min(args.steps, args.stop_after or args.steps)
    if start >= limit:
        raise ValueError('Requested endpoint already reached')
    wrapped = LossForward(model)
    if runtime.distributed:
        wrapped = DistributedDataParallel(wrapped, device_ids=[runtime.device.index] if runtime.device.type == 'cuda' else None,
                                         broadcast_buffers=False)
    args.output.mkdir(parents=True, exist_ok=True)
    if runtime.primary:
        atomic_json(args.output / 'recipe.json', recipe)
        atomic_json(args.output / 'data_profile.json', {'training': training.profile, 'validation': validation.profile})
    initial = evaluate_teacher(model, validation, runtime, args.batch_size)
    if runtime.primary:
        atomic_json(args.output / ('resume_validation.json' if args.resume else 'initial_validation.json'), initial)
    begun = time.monotonic()
    batch_seconds, update_seconds, update_count = 0., 0., 0
    model.train()
    for step in range(start, limit):
        epoch, offset = divmod(step, len(training) // global_batch)
        order = torch.randperm(len(training), generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()
        begin = offset * global_batch + runtime.rank * args.batch_size
        loaded_at = time.monotonic()
        batch = training.batch(order[begin:begin + args.batch_size], runtime.device)
        batch_seconds += time.monotonic() - loaded_at
        updated_at = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        losses = wrapped(batch)
        if not torch.isfinite(losses['total']):
            raise RuntimeError('Non-finite AR next-unit loss')
        losses['total'].backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        if float(norm) <= 0:
            raise RuntimeError('AR planner has no nonzero gradient')
        if args.check_only:
            audit = frozen_audit(model, frozen)
            audit.update(rank=runtime.rank, loss=float(losses['total'].detach()), gradient_norm=float(norm),
                         world_size=runtime.world_size, phase=args.phase)
            atomic_json(args.output / f'preflight_rank{runtime.rank}.json', audit)
            return
        optimizer.step()
        scheduler.step()
        if runtime.device.type == 'cuda':
            torch.cuda.synchronize(runtime.device)
        update_seconds += time.monotonic() - updated_at
        update_count += 1
        if runtime.primary and ((step + 1) % 10 == 0 or step + 1 == limit):
            progress = {'step': step + 1, 'steps': args.steps, 'loss': float(losses['total'].detach()),
                'gradient_norm': float(norm), 'elapsed_seconds': time.monotonic() - begun,
                'world_size': runtime.world_size, 'phase': args.phase}
            atomic_json(args.output / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
        if (step + 1) % args.evaluate_every == 0 or step + 1 == limit:
            metrics = evaluate_teacher(model, validation, runtime, args.batch_size)
            audit = frozen_audit(model, frozen)
            rng_states = runtime.gather_rng(generator)
            if runtime.primary:
                source_book = payload.get('metadata', {}).get('recovery_recipe', {}).get('codebook_sha256')
                checkpoint = model.checkpoint(step + 1, ar_recipe=recipe,
                    recovery_recipe={'codebook_sha256': source_book, 'experimental_ar': True})
                checkpoint.update(ar_step=step + 1, ar_frozen_hashes=frozen, optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), rank_rng_states=rng_states, world_size=runtime.world_size)
                atomic_save(checkpoint, args.output / 'last.pt')
                atomic_save(checkpoint, args.output / f'step_{step + 1:04d}.pt')
                atomic_json(args.output / f'audit_step_{step + 1:04d}.json', audit)
                result = {'step': step + 1, 'teacher_forced_validation': metrics,
                          'elapsed_seconds': time.monotonic() - begun,
                          'this_process_batch_loading_seconds': batch_seconds,
                          'this_process_optimizer_update_seconds': update_seconds,
                          'this_process_update_count': update_count}
                with (args.output / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(result) + '\n')
                print(json.dumps(result), flush=True)
            if runtime.distributed:
                torch.distributed.barrier()
            model.train()
    if runtime.primary and limit == args.steps:
        atomic_json(args.output / 'complete.json', {'steps': args.steps, 'phase': args.phase,
            'world_size': runtime.world_size, 'quality_passed': False, 'free_generation_evaluation_required': True})


@torch.inference_mode()
def evaluate(args, runtime):
    if runtime.distributed:
        raise ValueError('Explicit free-generation/audio evaluation uses one device')
    if args.output.exists():
        raise ValueError('Evaluation requires a new output directory')
    model, payload = load_ar_checkpoint(args.checkpoint)
    model.to(runtime.device).requires_grad_(False).eval()
    model.semantic_planner.configure_sampling(args.sampling, args.temperature, args.top_k, args.sampling_seed)
    data = SelectedSamples(args.manifest, args.selection, model, 'val', model.ar_phase, args.conditional_cache)
    before = frozen_hashes(model)
    report = {'checkpoint': str(args.checkpoint), 'checkpoint_sha256': file_hash(args.checkpoint),
        'architecture': AR_ARCHITECTURE, 'checkpoint_step': payload.get('ar_step'), 'phase': model.ar_phase,
        'manifest_sha256': file_hash(args.manifest), 'selection_sha256': file_hash(args.selection),
        'teacher_forced': evaluate_teacher(model, data, runtime, args.batch_size),
        'free_running': evaluate_free(model, data, runtime.device, args.free_count),
        'planner_sampling': dict(model.semantic_planner.sampling), 'acoustic_seed': args.seed,
        'test_split_read': False, 'production_candidate_promoted': False}
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / 'units_report.json', report)
    if args.audio_selection:
        if model.ar_phase != 'conditional':
            raise ValueError('Normal A-only audio requires conditional adaptation')
        from model.full_speech.codec import FrozenEncodec
        from scripts.diagnose_quality import recognize, save_wave
        audio_data = SelectedSamples(args.manifest, args.audio_selection, model, 'val', 'conditional')
        if len(audio_data) != 8 or not {row['path'] for row in audio_data.records} <= {row['path'] for row in data.records}:
            raise ValueError('Audio selection must be exactly eight selected validation conversations')
        codec = FrozenEncodec().to(runtime.device).eval()
        output = args.output / 'audio'
        output.mkdir(parents=True, exist_ok=True)
        audio_report = {key: value for key, value in report.items() if key not in ('teacher_forced', 'free_running')}
        audio_report.update(examples=[], asr_model='openai/whisper-base.en', acoustic_seed=args.seed,
            limitations=['Reference WER and ASR are not human listening or response relevance.',
                        'Predicted-length audio has no B input; reference and oracle-units paths are diagnostic.'])
        for index, row in enumerate(audio_data.records):
            batch = audio_data.batch([index], runtime.device)
            inputs = person_a_only(batch)
            generated = model.generate_batch(inputs, codec, seed=args.seed)
            normalized = (batch['semantic'] - model.semantic_mean) / model.semantic_std
            oracle = model.generate_batch(inputs, codec, seed=args.seed, oracle_semantic=normalized,
                                          oracle_duration=batch['duration'])
            entry = {'path': row['path'], 'conversation_id': row['conversation_id'], 'input_text': row['input_text'],
                     'reference_text': row['response_text'], 'paths': {}}
            waves = {'reference': batch['waveform'][0, :int(batch['waveform_len'][0])],
                     'oracle_units': oracle['waveform'][0, :int(oracle['audio_lengths'][0])],
                     'predicted_length_ar': generated['waveform'][0, :int(generated['audio_lengths'][0])]}
            for name, waveform in waves.items():
                entry['paths'][name] = {**save_wave(output, f'{index:02d}_{name}.wav', waveform.cpu().numpy()),
                    'normal_inference': name == 'predicted_length_ar'}
            audio_report['examples'].append(entry)
            atomic_json(output / 'generated_report.json', audio_report)
        model.to('cpu')
        del codec
        if runtime.device.type == 'cuda':
            torch.cuda.empty_cache()
        audio_report = recognize(audio_report, output, str(runtime.device))
        atomic_json(output / 'report.json', audio_report)
    report['frozen_audit'] = frozen_audit(model, before)
    atomic_json(args.output / 'units_report.json', report)
    print(json.dumps({'teacher_forced': report['teacher_forced'], 'free_running': report['free_running']['summary']}), flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'selection', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    for name in ('initialize', 'resume', 'checkpoint'):
        source.add_argument('--' + name, type=Path)
    parser.add_argument('--phase', choices=('prior', 'conditional', 'evaluate'), required=True)
    parser.add_argument('--steps', type=int, default=1600)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--evaluate-every', type=int, default=200)
    parser.add_argument('--lr', type=float, default=.0003)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--free-count', type=int, default=8)
    parser.add_argument('--audio-selection', type=Path)
    parser.add_argument('--conditional-cache', choices=('lazy', 'inputs_units'), default='lazy')
    parser.add_argument('--sampling', choices=('greedy', 'categorical'), default='greedy')
    parser.add_argument('--temperature', type=float, default=.8)
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--sampling-seed', type=int, default=42)
    args = parser.parse_args(argv)
    if (args.phase == 'evaluate') != bool(args.checkpoint):
        parser.error('Evaluate uses --checkpoint; adaptation uses --initialize or --resume')
    if min(args.steps, args.batch_size, args.evaluate_every, args.free_count) < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error('Positive training/evaluation limits required')
    if args.stop_after is not None and not 0 < args.stop_after <= args.steps:
        parser.error('--stop-after must be within the fixed update budget')
    if args.audio_selection and args.phase != 'evaluate':
        parser.error('--audio-selection is for explicit evaluation only')
    if not math.isfinite(args.temperature) or args.temperature <= 0 or args.top_k < 1:
        parser.error('Invalid categorical sampling settings')
    if args.phase != 'evaluate' and (args.sampling != 'greedy' or args.temperature != .8 or args.top_k != 20 or args.sampling_seed != 42):
        parser.error('Sampling options are for explicit evaluation only')
    return args


def main():
    args = parse_args()
    torch.set_num_threads(2)
    runtime = DistributedRuntime.initialize(args.device)
    try:
        (evaluate if args.phase == 'evaluate' else train)(args, runtime)
    finally:
        runtime.close()


if __name__ == '__main__':
    main()
