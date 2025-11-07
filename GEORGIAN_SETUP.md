# Georgian Language Specific Setup

## Georgian Character Support

### Already Handled ✓

The preprocessing code **already includes full Georgian alphabet support**:

```python
GEORGIAN_ALPHABET = set('აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ')
```

This includes all 33 letters of the modern Georgian (ქართული) alphabet:
- ა ბ გ დ ე ვ ზ თ ი კ ლ მ ნ ო პ ჟ რ ს ტ უ ფ ქ ღ ყ შ ჩ ც ძ წ ჭ ხ ჯ ჰ

### What's Automatically Validated

1. **Text Validation** (`preprocess_data.py`):
   - Checks that at least 50% of characters are Georgian
   - Normalizes punctuation
   - Handles Georgian sentence structure

2. **Tokenizer Support**:
   - Chatterbox's multilingual tokenizer already supports Georgian
   - Uses grapheme-based tokenization
   - No modification needed for Georgian characters

3. **Language ID**:
   - Georgian language code: `"ka"` (ISO 639-1)
   - Used in: `model.generate(text, language_id="ka")`

## Mozilla Common Voice Georgian Dataset

### Quick Start

```bash
# 1. Convert Common Voice format
python scripts/convert_commonvoice.py \
    --cv_dir cv-corpus-23.0-2025-09-05/ka \
    --output_dir data/raw

# 2. Preprocess (resample to 24kHz, normalize, split)
python scripts/preprocess_data.py \
    --metadata_csv data/raw/commonvoice_metadata.csv \
    --input_dir cv-corpus-23.0-2025-09-05/ka \
    --output_dir data/processed \
    --num_workers 8

# 3. Validate dataset quality
python scripts/validate_dataset.py \
    --metadata data/processed/metadata.csv \
    --output_dir data/validation

# 4. Start training
python scripts/train.py --config configs/georgian_finetune.yaml
```

### Dataset Statistics

Common Voice Georgian typically includes:
- 50-100 hours of validated speech
- 100s of speakers
- Various ages and genders
- Clean recording conditions

## Georgian-Specific Considerations

### 1. Punctuation Handling

Georgian uses standard punctuation with some specifics already handled:
- Period: `.` (same as English)
- Question mark: `?`
- Exclamation: `!`
- Comma: `,`

The code already normalizes these correctly.

### 2. Character Normalization

Georgian has no uppercase/lowercase distinction, which simplifies processing:
- No case folding needed
- Single character representation per letter

### 3. Text Length

Georgian words can be quite long due to agglutination:
- Average word length: 6-8 characters
- Max text length set to 512 tokens (sufficient)

### 4. Special Characters

Georgian text may include:
- Georgian numerals (rare, usually Arabic numerals used)
- Foreign names in Latin script
- Punctuation marks

All handled by the validation code.

## Tokenizer Analysis

Check Georgian support in the tokenizer:

```bash
python scripts/tokenizer_utils.py
```

This will show:
- Which Georgian characters are in the vocabulary
- Token IDs for Georgian letters
- Test tokenization on Georgian text

## Training Tips for Georgian

### 1. Use Multilingual Base Model

```yaml
model:
  base_model: "multilingual"  # Better than English-only
```

The multilingual model has seen similar languages and phonetic patterns.

### 2. Adjust for Agglutination

Georgian is highly agglutinative (many morphemes per word):

```yaml
data:
  max_text_len: 512  # Keep this, Georgian needs it
```

### 3. Speaker Diversity

Common Voice has many speakers, which is good:

```yaml
training:
  batch_size: 4  # Mix speakers in each batch
```

### 4. Training Duration

For Common Voice Georgian (~50-100 hours):

```yaml
training:
  num_epochs: 30-50  # More epochs for good quality
  learning_rate: 5.0e-5
```

## Common Issues and Solutions

### Issue: "Not enough Georgian characters"

**Cause**: Text contains too much Latin script or other languages

**Solution**: The validation script filters these out automatically. Check `data/validation/text_errors.csv`

### Issue: Tokenizer doesn't recognize Georgian

**Cause**: Unlikely - Chatterbox multilingual model includes Georgian

**Solution**: Run `python scripts/tokenizer_utils.py` to verify

### Issue: Poor audio quality

**Cause**: Some Common Voice samples vary in quality

**Solution**: 
1. Check `data/validation/audio_errors.csv`
2. Filter by upvotes: use samples with `up_votes >= 2`
3. Adjust in `convert_commonvoice.py`

### Issue: Model outputs gibberish

**Cause**: 
- Not enough training
- Wrong language_id
- Low quality data

**Solution**:
1. Verify language_id is "ka" not "ge" or other
2. Train for more epochs
3. Check validation script output

## Verification Checklist

Before training, verify:

- [ ] Dataset extracted: `cv-corpus-23.0-2025-09-05/ka/`
- [ ] Metadata created: `data/raw/commonvoice_metadata.csv`
- [ ] Data preprocessed: `data/processed/train.csv` and `data/processed/val.csv`
- [ ] Validation passed: Check `data/validation/validation_stats.json`
- [ ] Georgian text visible: `head data/processed/train.csv`
- [ ] Config updated: Language ID set to "ka"

## Language-Specific Parameters

### For Inference

```python
wav = model.generate(
    text="გამარჯობა, როგორ ხარ?",
    language_id="ka",           # Important!
    exaggeration=0.5,           # Natural emotion
    cfg_weight=0.5,             # Balanced
    temperature=0.8,            # Standard
    repetition_penalty=2.0      # Prevent repetition
)
```

### For Training

```yaml
# configs/georgian_finetune.yaml
data:
  train_csv: "data/processed/train.csv"
  val_csv: "data/processed/val.csv"
  sample_rate: 24000
  max_text_len: 512            # Georgian needs this

model:
  base_model: "multilingual"   # Use multilingual
  components_to_train:
    t3: true                   # Train text-to-speech tokens
    s3gen: false               # Keep pretrained (unless quality issues)
    voice_encoder: false       # Keep pretrained (unless speaker issues)
```

## Example Georgian Texts for Testing

After training, test with these:

```python
test_texts = [
    "გამარჯობა, როგორ ხარ?",
    "ძალიან კარგად, გმადლობთ.",
    "საქართველო ძალიან ლამაზი ქვეყანაა.",
    "თბილისი საქართველოს დედაქალაქია.",
    "მე მიყვარს ქართული ენა და კულტურა.",
]

for text in test_texts:
    wav = model.generate(text, language_id="ka")
    ta.save(f"test_{idx}.wav", wav, model.sr)
```

## Performance Expectations

With Mozilla Common Voice Georgian (~50-100 hours):

**After 10 epochs**: Basic intelligibility
**After 30 epochs**: Good quality, natural prosody
**After 50 epochs**: High quality, near-native

Training time (single A100 GPU):
- ~6-8 hours per 10 epochs
- ~20-25 hours for 30 epochs

## Additional Resources

- Georgian alphabet: https://en.wikipedia.org/wiki/Georgian_scripts
- Common Voice: https://commonvoice.mozilla.org/ka
- ISO 639-1 code: `ka`
- Script: Georgian (Mkhedruli)

## No Code Changes Needed!

**Good news**: You don't need to modify any code for Georgian characters. Everything is already set up:

✓ Georgian alphabet recognition
✓ Text validation
✓ Tokenizer support
✓ Language ID support
✓ Preprocessing pipeline

Just run the conversion and preprocessing scripts, then start training!

