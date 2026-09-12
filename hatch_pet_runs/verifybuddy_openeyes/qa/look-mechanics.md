# VerifyBuddy open-eyes look mechanics

VerifyBuddy is an adult-coded humanoid chibi sticker pet. Gaze direction should come from open eyes, pupils, eyelids, eyebrows, head/neck turn, and restrained upper-body follow-through. Keep both eyes open whenever both eyes are visible; in profile poses, the visible eye must remain open. Do not use winks, closed eyes, or squint-only expressions.

The cropped black-and-gold shirt, exposed non-sexual navel, neural-verification charm, dark pants, and black-gold shoes are part of the identity. Preserve the shorter shirt and visible navel in frontal and three-quarter poses. The charm stays attached to the shirt.

Cardinal pose families:

- `000 up`: frontal body, both eyes open, pupils and chin lift upward, brows rise.
- `090 screen-right`: open visible eye, nose, mouth, and head turn toward the viewer's right; torso follows slightly while the navel remains visible if the torso is not profile.
- `180 down`: frontal body, open eyes angle downward, chin lowers toward the cropped shirt/charm.
- `270 screen-left`: open visible eye, nose, mouth, and head turn toward the viewer's left; torso follows slightly while the navel remains visible if the torso is not profile.

Intermediate directions interpolate smoothly in 22.5-degree steps. Feet/lower body stay visually anchored. Do not rotate, skew, or warp the whole sprite to fake direction.
