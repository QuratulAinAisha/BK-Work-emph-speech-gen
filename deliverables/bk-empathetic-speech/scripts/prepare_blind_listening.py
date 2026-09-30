"""Prepare blank, blinded human listening forms. / 빈 블라인드 청취 평가 양식을 만듭니다."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil
import wave


RATING_FIELDS = ('intelligibility', 'relevance', 'emotional_appropriateness', 'naturalness')
CSV_FIELDS = ('package_id', 'listener_id', 'trial_id', *RATING_FIELDS, 'notes')


def file_hash(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def report_argument(value):
    label, separator, path = value.partition('=')
    if not separator or not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError('Use --report label=path/to/report.json')
    return label.strip(), Path(path.strip())


def report_variant_argument(value):
    label, separator, variant = value.partition('=')
    if not separator or not label.strip() or not variant.strip():
        raise argparse.ArgumentTypeError('Use --report-variant label=variant_name')
    return label.strip(), variant.strip()


def selected_variants(report_specs, default_variant, report_variants=None):
    labels = [label for label, _ in report_specs]
    if len(labels) != len({label.casefold() for label in labels}):
        raise ValueError('Report labels must be unique')
    if not isinstance(default_variant, str) or not default_variant.strip():
        raise ValueError('Default variant must be nonempty')
    if not report_variants:
        return {label: default_variant for label in labels}
    # A mixed package needs an explicit choice for every source. / 혼합 패키지는 모든 출처의 선택을 명시합니다.
    overrides = {}
    seen = set()
    for label, variant in report_variants:
        if label.casefold() in seen:
            raise ValueError('Duplicate report-variant label: ' + label)
        seen.add(label.casefold())
        if label not in labels:
            raise ValueError('Unknown report-variant label (labels must match exactly): ' + label)
        if not isinstance(variant, str) or not variant.strip():
            raise ValueError('Report variant must be nonempty: ' + label)
        overrides[label] = variant
    missing = set(labels) - set(overrides)
    if missing:
        raise ValueError('Report-variant overrides require exact label coverage; missing: ' + ', '.join(sorted(missing)))
    return overrides


def contained_wav(report_path, filename):
    relative = Path(filename)
    if relative.is_absolute() or relative.drive or '..' in relative.parts or relative.suffix.lower() != '.wav':
        raise ValueError('Audio must be a relative WAV path inside its report folder')
    base = report_path.parent.resolve()
    path = (base / relative).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        raise ValueError('Missing audio or audio path escapes its report folder')
    try:
        with wave.open(str(path), 'rb') as stream:
            frames, channels, rate, width = (stream.getnframes(), stream.getnchannels(),
                                             stream.getframerate(), stream.getsampwidth())
            if min(frames, channels, rate, width) < 1 or stream.getcomptype() != 'NONE':
                raise ValueError('Audio must contain nonempty uncompressed PCM samples')
            remaining = frames
            while remaining:
                chunk = min(65536, remaining)
                if len(stream.readframes(chunk)) != chunk * channels * width:
                    raise ValueError('Truncated WAV data')
                remaining -= chunk
    except (wave.Error, EOFError) as error:
        raise ValueError(f'Invalid PCM WAV: {path.name}') from error
    return path, {'frames': frames, 'channels': channels, 'sample_rate': rate,
                  'sample_width_bytes': width, 'seconds': frames / rate}


def read_sources(report_specs, variant, report_variants=None):
    if not report_specs:
        raise ValueError('At least one report is required')
    variants = selected_variants(report_specs, variant, report_variants)
    sources = []
    for label, path in report_specs:
        selected_variant = variants[label]
        path = path.resolve()
        raw = path.read_bytes()
        report = json.loads(raw)
        examples = {}
        for row in report.get('examples', []):
            identity = row.get('conversation_id')
            text = row.get('input_text')
            if not isinstance(identity, str) or not identity or identity in examples:
                raise ValueError('Conversation IDs must be nonempty and unique inside every report')
            if not isinstance(text, str) or not text.strip():
                raise ValueError('Every conversation needs A input text')
            if selected_variant not in row.get('paths', {}):
                raise ValueError(f'Report {label} is missing selected variant {selected_variant}')
            filename = row['paths'][selected_variant].get('file')
            if not isinstance(filename, str) or not filename:
                raise ValueError('Selected variant has no audio file')
            audio, details = contained_wav(path, filename)
            examples[identity] = {'input_text': text, 'audio_path': audio,
                'audio_sha256': file_hash(audio), 'audio_details': details,
                'source_example_path': row.get('path')}
        if not examples:
            raise ValueError('Reports must contain at least one conversation')
        sources.append({'label': label, 'selected_variant': selected_variant,
            'report_path': path, 'report_sha256': hashlib.sha256(raw).hexdigest(),
            'checkpoint': report.get('checkpoint'), 'reported_checkpoint_sha256': report.get('checkpoint_sha256'),
            'checkpoint_step': report.get('checkpoint_step', report.get('recovery_step')),
            'manifest_sha256': report.get('manifest_sha256'), 'selection_sha256': report.get('selection_sha256'),
            'examples': examples})
    expected = sources[0]['examples']
    for source in sources[1:]:
        if set(source['examples']) != set(expected):
            raise ValueError('Reports must contain identical conversation ID sets; no silent intersection')
        if any(source['examples'][key]['input_text'] != expected[key]['input_text'] for key in expected):
            raise ValueError('A input text differs between matched conversations')
    return sources


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Listen and rate</title><style>
:root{font-family:system-ui,sans-serif;color:#18232c;background:#f3f5f7;color-scheme:light}
body{margin:0}main{max-width:920px;margin:auto;padding:28px 20px 60px}h1{margin:0 0 12px;font-size:30px}
p{line-height:1.55}header,.trial{background:white;border:1px solid #dce3e8;border-radius:12px;padding:24px;margin-bottom:20px}
.muted{color:#4f606e}.toolbar{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin-top:18px}
button{background:#185d73;border:0;color:white;border-radius:8px;padding:12px 18px;font-size:16px;cursor:pointer}
input,select,textarea{font:inherit;border:1px solid #aab8c2;border-radius:6px;padding:8px;box-sizing:border-box}
label{display:flex;flex-direction:column;gap:7px}input{max-width:240px}audio{width:100%;margin:14px 0}
.context{white-space:pre-wrap;background:#f0f5f8;border-left:4px solid #5091a4;padding:14px;line-height:1.55}
.scores{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin:12px 0 18px}
select{width:100%;background:white}textarea{width:100%;min-height:65px;resize:vertical}
h2{font-size:19px;margin:0 0 12px}summary{cursor:pointer;font-weight:600}li{margin:8px 0;line-height:1.45}
small{font-size:13px}#status{font-weight:600}.trial-id{font-family:monospace;font-size:12px;color:#647582}
@media(max-width:580px){main{padding:16px 12px}.scores{grid-template-columns:1fr}header,.trial{padding:18px}}
</style></head><body><main><header><h1>Listen and rate</h1>
<p>Read what Person A said, listen to the response, then rate what you heard. Different wording can still be appropriate. Leave a score blank if you cannot judge it.</p>
<p class="muted">Person A is provided as text only; no Person A audio is available here. Judge emotional fit only against that provided context, without assuming A's vocal tone.</p>
<details open><summary>How to score</summary><ul>
<li><strong>Intelligibility:</strong> How easily can you understand the spoken words?</li>
<li><strong>Relevance:</strong> Does the response address Person A's situation?</li>
<li><strong>Emotional appropriateness:</strong> Does the response's tone fit the situation?</li>
<li><strong>Naturalness:</strong> Does pronunciation, rhythm and delivery sound natural?</li>
</ul><p>1 = very poor · 2 = poor · 3 = mixed · 4 = good · 5 = very good.</p></details>
<div class="toolbar"><label>Listener ID (optional)<input id="listener" autocomplete="off" placeholder="Use a nickname"></label>
<button id="export" type="button">Export ratings CSV</button><span id="status" role="status"></span></div>
<p class="muted"><small>Use a comfortable volume and the same headphones or speakers throughout. Scores are subjective judgments of these samples; they are not a validated empathy measure.</small></p>
<p class="muted"><small id="save-status">Export before closing. Your ratings remain on this device.</small></p></header>
<div id="trials"></div></main><script type="application/json" id="trials-data">__TRIAL_DATA__</script>
<script>
"use strict";
const data=JSON.parse(document.getElementById("trials-data").textContent);
const fields=["intelligibility","relevance","emotional_appropriateness","naturalness"];
const titles={intelligibility:"Intelligibility",relevance:"Relevance",emotional_appropriateness:"Emotional appropriateness",naturalness:"Naturalness"};
const storeKey="blind_listening_"+data.package_id;
const ratings=Object.fromEntries(data.trials.map(t=>[t.trial_id,Object.fromEntries([...fields,"notes"].map(f=>[f,""]))]));
const listener=document.getElementById("listener");
try{const saved=JSON.parse(localStorage.getItem(storeKey)||"null");if(saved){listener.value=typeof saved.listener==="string"?saved.listener:"";
for(const t of data.trials){const row=saved.ratings&&saved.ratings[t.trial_id];if(!row)continue;
for(const f of fields)if(["1","2","3","4","5"].includes(row[f]))ratings[t.trial_id][f]=row[f];
if(typeof row.notes==="string")ratings[t.trial_id].notes=row.notes;}}}catch(e){}
function progress(){const complete=data.trials.filter(t=>fields.every(f=>ratings[t.trial_id][f]!=="")).length;
document.getElementById("status").textContent=complete+" of "+data.trials.length+" fully rated";}
function persist(){try{localStorage.setItem(storeKey,JSON.stringify({listener:listener.value,ratings}));}
catch(e){document.getElementById("save-status").textContent="This browser cannot save between visits. Export before closing.";}progress();}
listener.addEventListener("input",persist);
data.trials.forEach((t,index)=>{const card=document.createElement("section");card.className="trial";
const heading=document.createElement("h2");heading.textContent="Trial "+(index+1);card.appendChild(heading);
const id=document.createElement("div");id.className="trial-id";id.textContent=t.trial_id;card.appendChild(id);
const intro=document.createElement("p");intro.textContent="Person A said:";card.appendChild(intro);
const context=document.createElement("div");context.className="context";context.textContent=t.input_text;card.appendChild(context);
const audio=document.createElement("audio");audio.controls=true;audio.preload="none";audio.src=t.audio;card.appendChild(audio);
const scores=document.createElement("div");scores.className="scores";
for(const f of fields){const label=document.createElement("label");label.textContent=titles[f];
const select=document.createElement("select");select.setAttribute("aria-label",titles[f]+" for trial "+(index+1));
for(const [value,text] of [["","Not rated"],["1","1 — Very poor"],["2","2 — Poor"],["3","3 — Mixed"],["4","4 — Good"],["5","5 — Very good"]]){
const option=document.createElement("option");option.value=value;option.textContent=text;select.appendChild(option);}
select.value=ratings[t.trial_id][f];select.addEventListener("change",()=>{ratings[t.trial_id][f]=select.value;persist();});
label.appendChild(select);scores.appendChild(label);}card.appendChild(scores);
const noteLabel=document.createElement("label");noteLabel.textContent="Notes (optional)";const note=document.createElement("textarea");
note.value=ratings[t.trial_id].notes;note.addEventListener("input",()=>{ratings[t.trial_id].notes=note.value;persist();});
noteLabel.appendChild(note);card.appendChild(noteLabel);document.getElementById("trials").appendChild(card);});
function csvCell(value){let text=String(value??"");if(/^[\s]*[=+@-]/.test(text))text="'"+text;return '"'+text.replace(/"/g,'""')+'"';}
document.getElementById("export").addEventListener("click",()=>{const columns=["package_id","listener_id","trial_id",...fields,"notes"];
const rows=[columns,...data.trials.map(t=>[data.package_id,listener.value,t.trial_id,...fields.map(f=>ratings[t.trial_id][f]),ratings[t.trial_id].notes])];
const blob=new Blob(["\ufeff"+rows.map(r=>r.map(csvCell).join(",")).join("\r\n")+"\r\n"],{type:"text/csv;charset=utf-8"});
const url=URL.createObjectURL(blob);const link=document.createElement("a");link.href=url;link.download="listening_ratings_"+data.package_id+".csv";
document.body.appendChild(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);});
progress();
</script></body></html>
'''


def prepare_package(report_specs, variant, output, seed=42, report_variants=None):
    output = output.resolve()
    if output.exists():
        raise ValueError('Use a new output directory; existing listening packages are never overwritten')
    sources = read_sources(report_specs, variant, report_variants)
    rng = random.Random(seed)
    # Bind saved ratings to these exact inputs. / 저장된 평가는 이 입력에만 연결합니다.
    identity = {'seed': seed, 'sources': [
        {'label': source['label'], 'selected_variant': source['selected_variant'],
         'report_sha256': source['report_sha256'],
         'audio_sha256': [source['examples'][key]['audio_sha256'] for key in sorted(source['examples'])]}
        for source in sources]}
    package_id = 'P' + hashlib.sha256(json.dumps(identity, sort_keys=True).encode('utf-8')).hexdigest()[:24]
    cases = [(source, identity) for identity in sorted(sources[0]['examples']) for source in sources]
    rng.shuffle(cases)
    public = {'package_id': package_id, 'trials': []}
    variants = {source['label']: source['selected_variant'] for source in sources}
    common_variant = next(iter(variants.values())) if len(set(variants.values())) == 1 else None
    private = {'package_id': package_id, 'seed': seed, 'variant': common_variant,
        'default_variant': variant, 'report_variants': variants,
        'variant_note': 'Each source/trial records its actual selected variant; top-level variant is null when mixed.',
        'rating_status': 'pending_human', 'human_ratings_collected': 0,
        'blinding_note': 'Share only the rater folder; keep this answer key with the coordinator.',
        'checkpoint_hash_note': 'Checkpoint hashes are reported provenance, not independently recomputed from model files.',
        'sources': [{key: str(value) if isinstance(value, Path) else value for key, value in source.items()
                     if key != 'examples'} for source in sources], 'trials': []}
    planned = []
    identifiers = set()
    for position, (source, identity) in enumerate(cases, 1):
        trial_id = 'T' + f'{rng.getrandbits(80):020x}'
        if trial_id in identifiers:
            raise RuntimeError('Opaque trial ID collision')
        identifiers.add(trial_id)
        row = source['examples'][identity]
        relative = f'audio/{trial_id}.wav'
        public['trials'].append({'trial_id': trial_id, 'input_text': row['input_text'], 'audio': relative})
        private['trials'].append({'trial_id': trial_id, 'presentation_order': position,
            'conversation_id': identity, 'source_label': source['label'],
            'source_variant': source['selected_variant'],
            'source_report': str(source['report_path']), 'source_report_sha256': source['report_sha256'],
            'source_checkpoint': source['checkpoint'], 'reported_checkpoint_sha256': source['reported_checkpoint_sha256'],
            'source_example_path': row['source_example_path'], 'source_audio': str(row['audio_path']),
            'source_audio_sha256': row['audio_sha256'], 'copied_audio_sha256': row['audio_sha256'],
            'public_audio': relative, 'audio_details': row['audio_details']})
        planned.append((row['audio_path'], relative, row['audio_sha256']))
    # Validate everything before creating the new package. / 새 패키지를 만들기 전에 모든 입력을 확인합니다.
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    rater = output / 'rater'
    (rater / 'audio').mkdir(parents=True)
    for source, relative, expected_hash in planned:
        destination = rater / relative
        shutil.copyfile(source, destination)
        if file_hash(destination) != expected_hash:
            raise RuntimeError('Source WAV changed while copying the listening package')
    # Only A text and opaque identifiers enter the rater page. / 평가 페이지에는 A 문장과 익명 ID만 넣습니다.
    encoded = json.dumps(public, ensure_ascii=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    (rater / 'index.html').write_text(HTML.replace('__TRIAL_DATA__', encoded), encoding='utf-8')
    with (rater / 'ratings_template.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({'package_id': package_id, 'trial_id': row['trial_id']} for row in public['trials'])
    (output / 'answer_key.json').write_text(json.dumps(private, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (rater / 'README.md').write_text(
        '# Listening instructions\n\nOpen index.html in a browser with the audio folder beside it. No network connection is needed. '
        'Use one comfortable volume setting and the same listening equipment throughout. Read Person A\'s text, '
        'listen, and choose your own scores. All ratings start blank. Leave unjudgeable dimensions blank.\n\n'
        '| Dimension | Judge |\n|---|---|\n'
        '| Intelligibility | How easily the spoken words can be understood. |\n'
        '| Relevance | Whether the response addresses A\'s situation. |\n'
        '| Emotional appropriateness | Whether the response\'s tone fits the situation. |\n'
        '| Naturalness | Pronunciation, rhythm and delivery. |\n\n'
        'Use 1 = very poor, 2 = poor, 3 = mixed, 4 = good, 5 = very good. '
        'Different wording can be appropriate. Judge each dimension separately. '
        'These are subjective sample ratings, not a validated empathy measure. Person A is provided as text only; '
        'no Person A audio is available here. Judge emotional fit only against that provided context, without assuming A\'s vocal tone.\n\n'
        'Enter an optional anonymous listener ID and export the CSV before closing. Partial exports preserve blank scores. '
        'Keep all trial IDs unchanged. If the page cannot be used, fill ratings_template.csv manually with 1–5 or blanks. '
        'Browser storage, when available, remembers only your own entries on this device.\n', encoding='utf-8')
    (output / 'README.md').write_text(
        '# Coordinator instructions\n\nHuman listening has not occurred: all generated scores are blank and pending. '
        'Share ONLY the rater folder. Keep answer_key.json and this coordinator folder away from raters until scoring is complete. '
        'Do not rename opaque audio files.\n\n'
        f'The package has {len(public["trials"])} trials covering {len(sources[0]["examples"])} matched conversations '
        f'and {len(sources)} sources. Presentation order is seeded and randomized; source labels are absent from the rater page and CSV. '
        'WAV files were copied byte for byte and their hashes verified. This preserves the exact submitted audio.\n\n'
        'Collect exported CSVs, verify ratings are 1–5 or blank, then join package_id and trial_id with the private answer key. '
        'Report listener counts, missing ratings and per-dimension results separately. '
        'Do not invent ratings or describe this package as completed listening evidence. '
        'Use independent raters where possible; the coordinator who saw labels is not blinded. '
        'A single seeded presentation order is not a counterbalanced study. '
        'These samples and subjective scores alone do not establish generalization or validated empathy.\n', encoding='utf-8')
    return {'output': str(output), 'rater_page': str(rater / 'index.html'), 'package_id': package_id,
            'conversations': len(sources[0]['examples']), 'trials': len(public['trials']),
            'rating_status': 'pending_human', 'human_ratings_collected': 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', action='append', type=report_argument, required=True)
    parser.add_argument('--variant', default='predicted_length_steps_8')
    parser.add_argument('--report-variant', action='append', type=report_variant_argument,
                        help='LABEL=VARIANT; when supplied, repeat for every report label exactly once.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare_package(args.report, args.variant, args.output, args.seed,
                                    report_variants=args.report_variant)), flush=True)


if __name__ == '__main__':
    main()
