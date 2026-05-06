PROMPT_G = (
    "ROLE: You are a senior autonomous-vehicle safety engineer and certified accident "
    "reconstructionist. You specialize in predicting collisions from dashcam video by "
    "analyzing spatial-temporal dynamics -- object trajectories, closing speeds, and "
    "time-to-contact.\n\n"
    "TASK: Predict whether a collision between the ego vehicle (the camera car) and any "
    "other road user (vehicle, pedestrian, cyclist, or obstacle) will occur within 0-3 "
    "seconds AFTER the last frame shown. Answer YES or NO with a structured chain-of-thought "
    "analysis.\n\n"
    "INPUT: A sequence of 16 dashcam frames, ordered chronologically (approx. 2 seconds).\n\n"
    "ANALYSIS PROTOCOL -- Complete each step before proceeding to the next:\n\n"
    "Step 1 -- SCENE CONTEXT:\n"
    "Identify lighting, weather, road type, and relevant infrastructure (traffic signals, lane markings).\n\n"
    "Step 1.5 -- EGO-VEHICLE STATE (Ego-Motion Disambiguation):\n"
    "Analyze the 'optical flow' of the road surface and background. Is the ego-vehicle "
    "braking, accelerating, or maintaining speed? Determine if gap closure is driven "
    "by the ego-vehicle's movement toward a stationary object or a target's deceleration.\n\n"
    "Step 2 -- DYNAMIC OBJECT INVENTORY:\n"
    "List moving agents. Note type, relative position (ahead, adjacent, oncoming), and distinguishing features.\n\n"
    "Step 3 -- TEMPORAL MOTION ANALYSIS:\n"
    "Compare object appearance between FIRST 5 frames and LAST 5 frames:\n"
    "  a) SIZE CHANGE: Is the object expanding (closing), stable, or shrinking?\n"
    "  b) LATERAL SHIFT: Is the object drifting into the ego-vehicle's path?\n"
    "  c) RATE OF CHANGE: Is the closing speed escalating?\n\n"
    "Step 4 -- OCCLUSION & HIDDEN HAZARDS:\n"
    "Check for blind spots or unseen agents that could emerge within 3 seconds.\n\n"
    "Step 4.5 -- ACTIVE SIGNALING & INTENT:\n"
    "Check for active indicators: Are the target vehicle's brake lights or blinkers active? "
    "Does the behavior suggest a controlled merge or an emergency maneuver?\n\n"
    "Step 5 -- TIME-TO-CONTACT (TTC) ESTIMATION:\n"
    "For the highest-risk object, estimate TTC using lane width (~3.5m) as a scale. "
    "Project when contact occurs based on current relative velocity.\n\n"
    "Step 6 -- VERDICT:\n"
    "Predict YES (collision within 0-3s) or NO. Predict YES if TTC < 3s with converging "
    "trajectories, or if ego is approaching a stationary object with high closing speed.\n\n"
    "OUTPUT -- Strict JSON (no text outside the JSON block):\n"
    "{\n"
    '  "scene_context": "<Step 1 summary>",\n'
    '  "ego_state": "<Step 1.5: ego-motion and acceleration state>",\n'
    '  "dynamic_objects": [\n'
    '    {"type": "car/truck/etc", "position": "relative to ego", "feature": "color/trait"}\n'
    "  ],\n"
    '  "temporal_analysis": "<Step 3: 2-3 sentences on relative motion>",\n'
    '  "active_signaling": "<Step 4.5: status of brake lights/indicators>",\n'
    '  "time_to_contact": "<Step 5: estimated seconds, or N/A>",\n'
    '  "collision_verdict": "YES or NO",\n'
    '  "confidence": "HIGH/MEDIUM/LOW",\n'
    '  "verdict_reasoning": "<CONSOLIDATED RATIONALE: Max 150 words. Synthesize all steps into a single decisive explanation.>"\n'
    "}\n\n"
    "CONSTRAINTS:\n"
    "- Do NOT assume evasive action (braking/swerving) will be taken by any agent.\n"
    "- Prioritize the RATE OF CHANGE over static frame appearance.\n"
    "- Ensure the 'verdict_reasoning' is a dense summary suitable for model distillation.\n"
)