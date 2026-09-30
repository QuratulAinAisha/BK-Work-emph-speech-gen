"""Compare fixed-checkpoint decoding without retraining. / 재학습 없이 고정 체크포인트 디코딩 비교."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from dataset.quality_speech_dataset import QualitySpeechDataset, collate_quality, person_a_only
from model.full_speech.units import UnitSpeechSystem
from model.full_speech.codec import FrozenEncodec
from model.full_speech.tensor_ops import mask_from_lengths, counts
from scripts.planner_sampling import sample_units
from scripts.analyze_planner_repetition import sequence_distribution
from scripts.evaluate_planner_controls import next_different_conversation, planner_inputs
from scripts.diagnose_quality import save_wave, recognize
from train_full import move_batch
from prepare_quality import atomic_json


VARIANTS = {
    'greedy_1': dict(mode='greedy', steps=1),
    'greedy_8': dict(mode='greedy', steps=8),
    'greedy_16': dict(mode='greedy', steps=16),
    'revisable_8': dict(mode='revisable', steps=8),
    'revisable_16': dict(mode='revisable', steps=16),
    'categorical_8_seed42': dict(mode='categorical', steps=8, seed=42, temperature=.8, top_k=20),
    'categorical_8_seed43': dict(mode='categorical', steps=8, seed=43, temperature=.8, top_k=20),
}


def chosen(data, selection):
    mapping = {r['path']: i for i, r in enumerate(data.records)}
    wanted = selection['val']
    if len(set(wanted)) != len(wanted) or any(p not in mapping for p in wanted):
        raise ValueError('Selection contains duplicates or wrong split')
    return [mapping[p] for p in wanted]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    for name in ('checkpoint', 'manifest', 'selection', 'audio-selection', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--units-only', action='store_true')
    parser.add_argument('--audio-only', action='store_true')
    args = parser.parse_args()
    if args.units_only and args.audio_only: parser.error('Choose one or neither')
    torch.set_num_threads(2)
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    selection = json.loads(args.selection.read_text()); audio_selection = json.loads(args.audio_selection.read_text())
    if any(s['manifest_sha256'] != manifest_hash for s in (selection, audio_selection)):
        raise ValueError('Manifest changed')
    args.output.mkdir(parents=True, exist_ok=True)
    model, _ = UnitSpeechSystem.from_checkpoint(args.checkpoint); model.to(args.device).eval()
    model.requires_grad_(False)
    data = QualitySpeechDataset(args.manifest, model.config, 'val')
    indices, audio_indices = chosen(data, selection), chosen(data, audio_selection)
    identity = {'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                'manifest_sha256': manifest_hash, 'selection_sha256': hashlib.sha256(args.selection.read_bytes()).hexdigest(),
                'audio_selection_sha256': hashlib.sha256(args.audio_selection.read_bytes()).hexdigest(),
                'variants': VARIANTS, 'production_weights_changed': False,
                'planner_memory_mode': getattr(model.config, 'planner_memory_mode', 'fused'),
                'length_and_acoustic_memory': 'original_fused'}
    old = args.output/'identity.json'
    if old.exists() and json.loads(old.read_text()) != identity: raise ValueError('Diagnostic recipe changed')
    atomic_json(old, identity)
    def load(index): return move_batch(collate_quality([data[index]]), torch.device(args.device))
    if not args.audio_only and not (args.output/'units_report.json').exists():
        donors = next_different_conversation([data.records[i] for i in indices])
        examples = []
        for position, index in enumerate(indices):
            batch, donor = load(index), load(indices[donors[position]])
            inputs, swapped = person_a_only(batch), person_a_only(donor)
            swapped.update(style_id=inputs['style_id'], speaker_id=inputs['speaker_id'])
            encoded = {'correct_a': model.encode_batch(inputs), 'shuffled_a': model.encode_batch(swapped)}
            style, _ = model.embeddings(inputs['style_id'], inputs['speaker_id'], 1)
            mask = mask_from_lengths(batch['semantic_len'])
            target = model.semantic_planner.codebook.encode((batch['semantic']-model.semantic_mean)/model.semantic_std)
            entry = {'conversation_id': data.records[index]['conversation_id'], 'frames': int(mask.sum()),
                     'target_unit_ids': target[mask].tolist(), 'conditions': {}}
            for context_name, enc in encoded.items():
                for name, settings in VARIANTS.items():
                    ids, trace = sample_units(model.semantic_planner, mask, style=style,
                                              **planner_inputs(model, enc), **settings)
                    entry['conditions'][context_name+'/'+name] = {'generated_unit_ids': ids[mask].tolist(),
                        'correct_units': int((ids[mask]==target[mask]).sum()), 'trace': trace}
            examples.append(entry)
            if (position+1)%16 == 0: print(json.dumps({'unit_cases':position+1,'total':len(indices)}),flush=True)
        total = sum(e['frames'] for e in examples)
        summary = {name: {'unit_accuracy':sum(e['conditions'][name]['correct_units'] for e in examples)/total,
                         **sequence_distribution([e['conditions'][name]['generated_unit_ids'] for e in examples])}
                   for name in examples[0]['conditions']}
        atomic_json(args.output/'units_report.json', {**identity,'count':len(examples),
            'oracle_length':True,'summary':summary,'examples':examples,
            'limitation':'True B length is used only for unit diagnostics; diversity is not relevance.'})
    if not args.units_only and not (args.output/'audio'/'report.json').exists():
        output=args.output/'audio'; output.mkdir(exist_ok=True)
        codec=FrozenEncodec().to(args.device).eval()
        report={**identity,'checkpoint':str(args.checkpoint),'examples':[],
            'normal_variants_use_only_a':True,'shared_acoustic_seed':42,
            'limitations':['Oracle B units and duration are a separately labeled control.',
                'Every variant uses the same A-predicted duration and acoustic noise.',
                'Categorical planner RNG is separate from the acoustic RNG.',
                'ASR is not human listening; reference WER does not score free-reply relevance.']}
        for index in audio_indices:
            batch=load(index);inputs=person_a_only(batch);enc=model.encode_batch(inputs)
            style,_=model.embeddings(inputs['style_id'],inputs['speaker_id'],1)
            duration,_=model.length_predictor(enc['context'],enc['affect'],enc['context_mask'],style)
            mask=mask_from_lengths(counts(duration,model.config.semantic_hz))
            entry={k:data.records[index][k] for k in ('conversation_id','input_text','response_text')}
            entry['reference_text']=entry.pop('response_text');entry['paths']={};entry['unit_traces']={}
            entry['paths']['reference']=save_wave(output,f'{index:06d}_reference.wav',batch['waveform'][0].cpu().numpy())
            oracle=model.generate_batch(inputs,codec,seed=42,
                oracle_semantic=(batch['semantic']-model.semantic_mean)/model.semantic_std,
                oracle_duration=batch['duration'])
            entry['paths']['oracle_units']=save_wave(output,f'{index:06d}_oracle_units.wav',
                oracle['waveform'][0,:int(oracle['audio_lengths'][0])].cpu().numpy())
            for name,settings in VARIANTS.items():
                ids,trace=sample_units(model.semantic_planner,mask,style=style,
                                       **planner_inputs(model,enc),**settings)
                # These supplied units are predicted from A, not B hints. / 제공 단위는 B 힌트가 아니라 A로 예측합니다.
                generated=model.generate_batch(inputs,codec,seed=42,
                    oracle_semantic=model.semantic_planner.codebook.centers[ids],oracle_duration=duration)
                entry['paths'][name]=save_wave(output,f'{index:06d}_{name}.wav',
                    generated['waveform'][0,:int(generated['audio_lengths'][0])].cpu().numpy())
                entry['unit_traces'][name]=trace
                if name=='greedy_8':
                    model.config.semantic_steps=8
                    normal=model.generate_batch(inputs,codec,seed=42)
                    if not torch.equal(normal['waveform'],generated['waveform']):
                        raise AssertionError('Greedy baseline audio differs from production path')
            report['examples'].append(entry)
            print(json.dumps({'audio_cases':len(report['examples']),'total':len(audio_indices)}),flush=True)
        del codec,model
        torch.cuda.empty_cache() if args.device.startswith('cuda') else None
        report=recognize(report,output,args.device);atomic_json(output/'report.json',report)
        print(json.dumps(report['summary']),flush=True)


if __name__=='__main__':main()
