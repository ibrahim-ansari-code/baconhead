One-Day Roblox Obby Agent — Full Plan
The Big Picture
You are building a three-layer system that mimics how a human plays: eyes see the screen, brain decides what to do, hands execute the action. The three layers map directly to that:

Perception — the eyes (OpenCV processing first-person screenshots)
High-level planner — the brain (Gemini deciding what to do next)
Low-level controller — the hands (a trained CNN executing precise key presses)

Everything runs in a continuous loop, roughly 20 times per second for the low level and once every 1.5–2 seconds for the high level.

Pre-Day Preparation
Choose your obby carefully. Pick one specific section — 5 to 10 platforms, no moving parts, no timed obstacles. A section you can complete yourself reliably in under 30 seconds. This scoping is critical — you are not building a general agent today, you are proving the architecture works on one constrained problem.
Set up Roblox correctly before anything else:

Use a throwaway account, not your main — automating gameplay violates Roblox's Terms of Service regardless of method, so keep risk contained to a disposable account
Switch to first-person view and lock it there — this eliminates the character detection problem entirely
Set camera sensitivity to a fixed value and don't touch it again — consistency between sessions is essential for the CNN
Set graphics to a medium-low setting — reduces visual noise and makes perception more reliable
Position the Roblox window at a fixed location on your screen and keep it there all day

Install all dependencies before the day starts so you're not troubleshooting package issues during your build time.

Hour 1–2: Screen Capture Pipeline
What you're building: A reliable way to grab the Roblox window as a numpy array of pixels that every other component will use.
What to understand:
The capture needs to be fast (under 5ms per frame), consistent (same region every time), and produce frames in a format your CV and ML libraries expect. You'll capture at full resolution for perception and Gemini, and at a small resolution (84×84) for the CNN — the CNN doesn't need fine detail, just enough visual information to distinguish platform from void.
What to verify before moving on:
Open a live preview window showing your capture output. Confirm it shows the Roblox game cleanly, is properly cropped to just the game window (no taskbar, no other apps bleeding in), and updates smoothly. If this isn't clean, nothing downstream will work.

Hour 2–3: Perception Layer
What you're building: A set of functions that take a raw screenshot and return structured information about the scene — specifically whether you're near an edge, whether the next platform is left/right/ahead, and whether you've died.
What to understand:
Since you're in first person and the obby has rainbow/multicolor platforms, color detection for platforms is unreliable. Your strategy flips this around entirely:
Void detection is your anchor. The void (the empty space you fall into) has a consistent color throughout the entire obby — it doesn't change with the rainbow theme. Everything that is not void is either a platform, your UI, or the sky. This is your primary signal.
Three regions of the first-person frame matter:

Bottom-center strip — this tells you your ground state. High void ratio here means you're near an edge or already falling. Low void ratio means you're safely on a platform.
Left and right halves — comparing platform mass between left and right tells you which direction the next platform is relative to where you're facing.
Upper portion — mostly irrelevant for a simple obby but can help Gemini understand the broader scene context.

Death detection works differently from platform detection. In first person, death in Roblox causes a specific visual event — the screen either fades, shows a respawn UI, or goes dark. You detect this by checking for that UI's distinctive colors or by checking if the overall scene has changed dramatically from the previous frame. A void ratio above 80–90% of the entire screen is also a strong death signal.
What to verify before moving on:
Run perception on a live frame and print the output. Walk your character to a platform edge and confirm "near edge" triggers. Die deliberately and confirm death detection fires. This verification is non-negotiable — bad perception poisons everything downstream.

Hour 3–4: Data Collection
What you're building: A dataset of (screenshot, action) pairs from you playing the obby in first person.
What to understand:
Behavioral cloning — the learning technique you're using — is only as good as its demonstrations. The model will literally try to copy what you do. This means:
Quality matters more than quantity — 25 clean, consistent runs where you play well beats 50 sloppy runs. Play at a slightly slower pace than normal. Deliberate, clear movements translate better to training signal.
Your action space is four things — move forward, move left, move right, and jump. Every frame of your recording captures which of these you're pressing at that exact moment. The dataset is a sequence of these (frame, action) pairs.
Record failure recovery too — intentionally go to the edge of a platform a few times and recover. The model needs to see what "nearly falling" looks like and what the correct response is, otherwise it will have no behavior for that situation.
Recording consistency checklist:

Same starting position every run
Same camera sensitivity (don't touch it)
Same graphics settings
First person locked throughout
25–30 runs minimum
About 45–60 minutes of total playtime

The resulting dataset will have tens of thousands of frames. This is your most valuable asset — treat it carefully and save it backed up.

Hour 4–6: Training the Behavioral Cloning Model
What you're building: A CNN that takes an 84×84 screenshot and outputs which action to take.
What to understand:
Why a CNN: Convolutional neural networks are specifically designed for image data. They learn spatial patterns — the relationship between where things are in the frame — which is exactly what you need. The first layers detect low-level features like edges and color gradients. Deeper layers combine those into higher-level patterns like "platform surface ahead" or "void below."
The architecture in plain English:

Several convolutional layers progressively compress the image while extracting features
A flatten operation converts the spatial feature map into a vector
Fully connected layers translate that vector into probabilities for each of the four actions
The model outputs whichever action has the highest probability

Training process:

Split your data 85% training, 15% validation — the validation set tells you if the model is actually learning to generalize or just memorizing
Train for about 30 epochs — one pass through all training data per epoch
Watch validation accuracy — above 70% means the model has genuinely learned something useful
Save the best model (lowest validation loss) not the last one — the last one may have started overfitting

What overfitting looks like: Training accuracy keeps climbing but validation accuracy plateaus or drops. This means the model is memorizing your specific runs rather than learning general patterns. Fix it with dropout layers (randomly zeroing activations during training) or by collecting more varied demonstration data.
What to verify before moving on:
Validation accuracy above 70%. If it's below 60%, stop and collect more demonstration data rather than continuing to train — more compute won't fix a data problem.

Hour 6–7: Building and Deploying the Two-Tier Agent
What you're building: The full agent loop that combines perception, Gemini, and the CNN into a running system.
What to understand:
The two-tier timing design is the most important architectural decision:
The Gemini API takes 400–800ms to respond. That's too slow to make every decision. But the CNN runs locally in under 5ms, so it can make decisions at 20 frames per second. You use both by giving them different jobs on different schedules:

Gemini runs every 1.5–2 seconds — it looks at the full screenshot and produces a high-level instruction like "the next platform is ahead and to the left, a jump will be needed." This updates the agent's current goal.
The CNN runs every 50ms — it takes the current screenshot and outputs the immediate action. It doesn't know about Gemini's plan directly — it just reacts to what it sees.

How they interact: Gemini's output can optionally be used to bias or filter the CNN's output. For example if Gemini says "jump needed" and the CNN outputs "move forward," you can weight the jump action higher. In practice for a simple obby the CNN alone may be sufficient — Gemini adds a safety net for confusing situations.
The death handling loop needs to be rock solid:

Detect death from perception
Release all held keys immediately (important — don't leave W held down through a death)
Wait for the respawn animation
Press the respawn confirmation
Wait for the character to fully load back at the checkpoint
Resume the main loop

What to verify before moving on:
Run the agent for 5 minutes and watch it. It will fail, but confirm that death detection fires correctly, the respawn sequence works, and the agent restarts cleanly each time. A clean death/respawn cycle is essential for the iteration phase.

Hour 7–8: Observe, Diagnose, and Iterate
What you're doing: Watching where the agent fails, understanding why, and fixing it with targeted data collection rather than more training time.
What to understand:
The agent will almost certainly fail at one or two specific spots consistently. This is valuable information — it tells you exactly what the model hasn't learned yet.
The diagnostic process:
Note exactly which platform the agent fails on and what it does wrong. The failure mode tells you what's missing from your training data:

Falls off the same edge repeatedly — you didn't record enough demonstrations of that specific visual scenario. Record 5–10 more runs that specifically approach that platform correctly.
Doesn't jump when it should — your demonstrations had too few jump examples at that point. Record targeted demonstrations with explicit jumps at that spot.
Runs in the wrong direction — perception may be giving wrong left/right signals. Check your void detection output at that frame visually.

The iteration loop:
Record targeted demonstrations → add to existing dataset → retrain for 10 epochs (not 30, just fine-tuning) → redeploy → observe again. Each cycle takes about 20–30 minutes. You can do 2–3 cycles in this final hour.

End of Day Success Criteria
WhatRealistic targetScreen capture workingHour 1Perception correctly detecting void and edgesHour 225+ first-person demonstrations recordedHour 4Model trained with validation accuracy above 70%Hour 6Agent running continuously with clean death/respawnHour 7Section completed successfully 50%+ of attemptsHour 8

The Most Important Thing to Internalize
The agent doesn't need to understand the game the way you do. It doesn't know what an obby is, what Roblox is, or what jumping means. It learns a mapping from pixels to key presses that happens to result in completing the section. That's it. Every architectural decision — first person view, void detection, small CNN, behavioral cloning — is in service of making that mapping as clean and learnable as possible from the data you can realistically collect in one day.