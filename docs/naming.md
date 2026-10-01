# Paper-Aligned Naming

| Paper term | Notation | Public field or filename |
| --- | --- | --- |
| Storyboard | B | `storyboard.json`, ordered `shots` array |
| Sketch | S_k | `sketch`, `sketches/shot_001.png` |
| Appearance prompt | A_k | `appearance_prompt` |
| Motion prompt | M_k | `motion_prompt` |
| Enhanced appearance prompt | enhanced A_k | `enhanced_appearance_prompt` |
| Conversion prompt | C_k^i | `conversion_prompt` |
| Structured dynamic prompt | D_k^i | `dynamic_prompt` |
| Colored reference keyframe | I_k^0 | `reference_keyframe.png` |
| Derivative keyframe | I_k^i | `derivative_keyframe_01.png` ... `05.png` |
| Local video clip | V_k^i | `local_clip_01.mp4` ... `05.mp4` |
| Composed shot | k-th shot | `shot_video.mp4` |
| Composed storyboard | final multi-shot video | `storyboard_video.mp4` |

The dynamic prompt has five components: `positive`, `animation_action`, `body_motion`, `facial_expression_change`, and `style_consistency`.

All five derivative keyframes use `reference_keyframe.png` as their image reference. Local clip 1 connects the reference to derivative 1; clip i connects derivative i-1 to derivative i. Shot order is the input array order, not filesystem traversal order.

Dataset import preserves original sample identifiers in `source_sample_id`, while public generation files use the names above. The downloaded dataset retains its original folder names; the import utility maps `static_prompt` to `appearance_prompt` and `story` to `motion_prompt` without changing the text.

All example and generation-manifest paths are relative to their containing JSON. Model files are selected by ComfyUI loader names rather than machine-specific absolute paths.
