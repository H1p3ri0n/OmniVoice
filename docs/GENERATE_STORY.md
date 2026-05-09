# OmniVoice Generation Notes

Practical notes on using OmniVoice's generation features.

---

## Inserting Pauses

There is no dedicated pause tag (`[pause]`, SSML `<break>`, etc.). Pauses are controlled via punctuation, which gets a `0.5` duration weight in the duration estimator ([`omnivoice/utils/duration.py:61`](../omnivoice/utils/duration.py)).

| Syntax | Effect |
|--------|--------|
| `,` or `，` | Short breath/pause |
| `.` or `。` | Sentence-end pause |
| `…` or `...` | Longer hesitation pause |
| Multiple punctuation `...` / `。。。` | Even longer (duration weights stack) |

> **Note:** Post-processing (`remove_silence`) is on by default and may trim long silences.  
> Pass `postprocess_output=False` to `generate()` if pauses are being removed unintentionally.
