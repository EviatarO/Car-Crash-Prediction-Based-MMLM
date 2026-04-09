SCORE_TOKEN = "<SCORE>"

BASE_PROMPT = (
    "You are analyzing a driving video clip. "
    "Provide a brief risk assessment for the FINAL frame.\n"
    "[Reasoning]:\n"
    f"[Risk Score]: {SCORE_TOKEN}\n"
)

TEACHER_PROMPT = (
    "You are analyzing a driving video clip. "
    "Use the provided target risk score and justify it with visual cues.\n"
    "Output strictly in this format:\n"
    "[Reasoning]: <short explanation>\n"
    "[Risk Score]: <float 0.0-1.0>\n"
)

TEACHER_BLIND_PROMPT = (
    "You are an expert driving safety analyst. "
    "You are given a sequence of 16 dashcam frames from a driving video, "
    "ordered chronologically (oldest to newest). "
    "The LAST frame represents the current moment.\n\n"
    "Analyze the sequence and determine whether a collision is imminent.\n\n"
    "Output STRICTLY in this format (no other text):\n"
    "[Collision_Reasoning]: <2-3 sentences explaining what you observe "
    "and why a collision is or is not about to happen>\n"
    "[Score]: <a single float between 0.0 and 1.0, where 0.0 = completely safe "
    "and 1.0 = collision is certain>\n"
)

PROMPT_A = (
    "You are an expert defensive driving instructor reviewing dashcam footage. "
    "You are given a sequence of 16 dashcam frames from a driving video, "
    "ordered chronologically (oldest to newest). "
    "The LAST frame represents the current moment.\n\n"
    "Your task is to PREDICT whether a collision will occur within the next "
    "2-5 seconds based on the trajectories visible in these frames.\n\n"
    "CRITICAL RULES:\n"
    "1. Assume other drivers are distracted and may NOT yield, brake, or swerve.\n"
    "2. If any vehicle, pedestrian, or obstacle is on a converging path with the "
    "camera vehicle, treat it as HIGH RISK regardless of current speed.\n"
    "3. Do NOT describe what happened in the past. Instead, extrapolate the current "
    "trajectories forward and predict where each vehicle will be in 2-3 seconds.\n"
    "4. Err on the side of caution: a missed collision is far worse than a false alarm.\n\n"
    "First provide your reasoning, then give your final answer.\n\n"
    "Output STRICTLY in this format (no other text):\n"
    "[Collision_Reasoning]: <2-3 sentences explaining the projected trajectories "
    "and why they will or will not intersect within the next few seconds>\n"
    "[Verdict]: <Answer with a single word: YES or NO>\n"
)

PROMPT_B = (
    "You are a traffic accident investigator reviewing dashcam evidence. "
    "You are given a sequence of 16 dashcam frames from a driving video, "
    "ordered chronologically (oldest to newest). "
    "The LAST frame represents the current moment.\n\n"
    "IMPORTANT CONTEXT: This footage was submitted as part of a collision report. "
    "A collision occurs shortly after the last frame shown. Your job is to identify "
    "the warning signs visible in these frames that indicate the collision is about "
    "to happen.\n\n"
    "Focus on:\n"
    "- Which vehicles or objects are involved in the developing collision\n"
    "- The closing speed and converging trajectories between them\n"
    "- Any signs of driver inattention, failure to brake, or dangerous maneuvers\n\n"
    "Based on your analysis, determine: are clear warning signs of an imminent "
    "collision visible in these frames?\n\n"
    "Output STRICTLY in this format (no other text):\n"
    "[Collision_Reasoning]: <2-3 sentences identifying the specific warning signs "
    "and the vehicles/objects that are about to collide>\n"
    "[Verdict]: <Answer with a single word: YES or NO>\n"
)

PROMPT_C = (
    "You are a computer vision engineer specializing in vehicle dynamics analysis. "
    "You are given a sequence of 16 dashcam frames from a driving video, "
    "ordered chronologically (oldest to newest). "
    "The LAST frame represents the current moment.\n\n"
    "Perform the following analysis step by step:\n"
    "1. Identify the nearest vehicle or obstacle ahead or approaching from the sides.\n"
    "2. Estimate the approximate distance to that object in the FIRST frame and the "
    "LAST frame (in meters, using lane width ~3.5m as reference).\n"
    "3. Calculate the approximate closing speed based on how that distance changed "
    "across the 16 frames (the sequence spans roughly 2 seconds).\n"
    "4. Estimate the time-to-contact: if the closing speed continues, how many "
    "seconds until the vehicles make contact?\n\n"
    "Based on your time-to-contact estimate, determine: will a collision occur "
    "within the next 3 seconds?\n\n"
    "Output STRICTLY in this format (no other text):\n"
    "[Collision_Reasoning]: <2-3 sentences with your distance estimates, closing "
    "speed estimate, and time-to-contact calculation>\n"
    "[Verdict]: <Answer with a single word: YES or NO>\n"
)

PROMPT_D = (
    "You are an autonomous vehicle safety system making a real-time decision. "
    "You are given a sequence of 16 dashcam frames from a driving video, "
    "ordered chronologically (oldest to newest). "
    "The LAST frame represents the current moment.\n\n"
    "You must make an EMERGENCY BRAKING decision. Answer this question:\n"
    "\"Will this vehicle collide with another vehicle, pedestrian, or obstacle "
    "within the next 3 seconds if no evasive action is taken?\"\n\n"
    "DECISION RULES:\n"
    "- If ANY object is getting closer frame-by-frame and is now within a "
    "dangerous distance, answer YES.\n"
    "- If vehicles are merging into the same lane space, answer YES.\n"
    "- If you are uncertain but see risk factors, answer YES -- false positives "
    "are safer than false negatives. A missed collision costs lives.\n"
    "- Only answer NO if the road ahead is clearly safe with no converging objects.\n\n"
    "Output STRICTLY in this format (no other text):\n"
    "[Collision_Reasoning]: <2-3 sentences explaining what you observe and your "
    "emergency braking decision with justification>\n"
    "[Verdict]: <Answer with a single word: YES or NO>\n"
)

PROMPT_E = (
    "You are a motion analysis expert reviewing dashcam footage for collision risk. "
    "You are given a sequence of 16 dashcam frames from a driving video, "
    "ordered chronologically (oldest to newest). "
    "The LAST frame represents the current moment.\n\n"
    "Perform a TEMPORAL COMPARISON analysis:\n"
    "1. Look at FRAME 1 (the oldest). Note the position and size of the nearest "
    "vehicle or obstacle relative to the camera.\n"
    "2. Look at FRAME 16 (the newest/current). Note the same object's position and "
    "size now.\n"
    "3. Describe the CHANGE: Did the object grow larger in the frame (getting closer)? "
    "Did it shift laterally (entering your lane)? Did it appear suddenly?\n"
    "4. Based on the RATE of change across these 16 frames (~2 seconds), project "
    "forward: will the object reach the camera vehicle within the next 2-3 seconds?\n\n"
    "IMPORTANT: If the nearest object has grown significantly larger between frame 1 "
    "and frame 16, this indicates rapid closing -- answer YES.\n\n"
    "Output STRICTLY in this format (no other text):\n"
    "[Collision_Reasoning]: <2-3 sentences comparing frame 1 vs frame 16, describing "
    "the rate of change, and predicting whether contact will occur>\n"
    "[Verdict]: <Answer with a single word: YES or NO>\n"
)

PROMPT_F = (
    "System Role: You are a Vision-Language Physics Engine. "
    "Your task is to analyze a 16-frame dashcam sequence and determine the causal "
    "path toward a potential collision within a 3-second horizon.\n\n"
    "Task: Analyze the temporal evolution of the scene.\n\n"
    "Analysis Steps (Internal Monologue):\n"
    "1. Depth/Scale: Is the bounding box of any object expanding rapidly? "
    "(Indicates high closing speed).\n"
    "2. Occlusion: Are there hidden zones (e.g., behind a parked truck) where "
    "a pedestrian or vehicle might emerge?\n"
    "3. Vectors: Are the trajectories of other agents converging with the ego-path?\n\n"
    "Output Format (Strict JSON):\n"
    "{\n"
    '  "environmental_context": "<brief description of weather, light, and road type>",\n'
    '  "dynamic_observations": [\n'
    '    {"object": "<type>", "motion": "closing/steady/receding", '
    '"lateral_intent": "merging/staying/crossing"}\n'
    "  ],\n"
    '  "causal_reasoning": "<2 sentences linking the motion of specific objects '
    'to a future collision or safety outcome>",\n'
    '  "collision_verdict": "<YES or NO>"\n'
    "}\n\n"
    "Constraint: Do not be safe. Be accurate. Base your verdict on the laws of physics "
    "and spatial proximity observed in the final 5 frames compared to the first 5 frames.\n"
)

PROMPT_G = (
    "ROLE: You are a senior autonomous-vehicle safety engineer and certified accident "
    "reconstructionist. You specialize in predicting collisions from dashcam video by "
    "analyzing spatial-temporal dynamics -- object trajectories, closing speeds, and "
    "time-to-contact.\n\n"
    "TASK: Predict whether a collision between the ego vehicle (the camera car) and any "
    "other road user (vehicle, pedestrian, cyclist, or obstacle) will occur within 0-3 "
    "seconds AFTER the last frame shown. Answer YES or NO with a structured chain-of-thought "
    "analysis.\n\n"
    "INPUT: A sequence of 16 dashcam frames, ordered chronologically:\n"
    "  - Frame 1 (oldest) through Frame 16 (most recent = current moment)\n"
    "  - The sequence spans approximately 2 seconds (~30 fps sampling)\n"
    "  - All frames are from the ego vehicle's forward-facing camera\n\n"
    "ANALYSIS PROTOCOL -- Complete each step before proceeding to the next:\n\n"
    "Step 1 -- SCENE CONTEXT:\n"
    "Identify the lighting (day/night/twilight), weather (clear/rain/fog/snow), "
    "road type (highway/urban/intersection/residential/parking), and relevant "
    "infrastructure (traffic signals, lane markings, barriers, crosswalks).\n\n"
    "Step 2 -- DYNAMIC OBJECT INVENTORY:\n"
    "List every moving or potentially moving agent in the scene. For each, note:\n"
    "  - Type (car, truck, bus, motorcycle, pedestrian, cyclist)\n"
    "  - Approximate position relative to ego (ahead-same-lane, ahead-adjacent-lane, "
    "approaching-from-left, approaching-from-right, oncoming, crossing)\n"
    "  - Color or distinguishing feature for tracking across frames\n\n"
    "Step 3 -- TEMPORAL MOTION ANALYSIS (most critical step):\n"
    "For each object identified in Step 2, compare its appearance between the FIRST 5 "
    "frames (1-5) and the LAST 5 frames (12-16):\n"
    "  a) SIZE CHANGE: Is the object's apparent size expanding rapidly "
    "(= closing fast), expanding slowly, stable, or shrinking (= receding)?\n"
    "  b) LATERAL SHIFT: Is the object drifting toward the ego vehicle's lane center?\n"
    "  c) ACCELERATION: Is the rate of change increasing (danger escalating) or "
    "decreasing (situation resolving)?\n\n"
    "Step 4 -- OCCLUSION & HIDDEN HAZARDS:\n"
    "Check for blind spots: areas behind parked vehicles, large trucks, or around "
    "curves where an unseen agent could suddenly appear. Note any partial objects at "
    "frame edges.\n\n"
    "Step 5 -- TIME-TO-CONTACT ESTIMATION:\n"
    "For the highest-risk object from Step 3, estimate the approximate time-to-contact "
    "using the lane width (~3.5m) as a visual ruler. If the object grew from roughly X% "
    "to Y% of frame width across 2 seconds, project when it would reach contact.\n\n"
    "Step 6 -- VERDICT:\n"
    "Predict YES (collision will occur within 0-3 seconds) or NO. Predict YES when ANY "
    "of these hold:\n"
    "  - Time-to-contact < 3s with converging trajectories\n"
    "  - A vehicle is actively merging into ego's lane with insufficient gap\n"
    "  - A pedestrian/cyclist is crossing the ego path at close range\n"
    "  - Ego is approaching a stationary/slow object with high closing speed\n\n"
    "OUTPUT -- Strict JSON (no text outside the JSON block):\n"
    "{\n"
    '  "scene_context": "<Step 1: lighting, weather, road type, infrastructure>",\n'
    '  "dynamic_objects": [\n'
    '    {"type": "<car/truck/pedestrian/cyclist>", "position": "<relative to ego>", '
    '"feature": "<color or distinguishing trait>"}\n'
    "  ],\n"
    '  "temporal_analysis": "<Step 3: 2-3 sentences comparing early vs late frames>",\n'
    '  "occlusion_check": "<Step 4: 1 sentence, or NONE>",\n'
    '  "time_to_contact": "<Step 5: estimated seconds, or N/A>",\n'
    '  "collision_verdict": "YES or NO",\n'
    '  "confidence": "HIGH or MEDIUM or LOW",\n'
    '  "verdict_reasoning": "<1-2 sentences: the decisive factor>"\n'
    "}\n\n"
    "CONSTRAINTS:\n"
    "- Analyze ONLY what is visible in the frames. Do not invent objects or events.\n"
    "- Do NOT assume evasive action will be taken -- predict based on current trajectories.\n"
    "- Prioritize accuracy over caution. A false alarm is equally costly as a missed "
    "collision.\n"
    "- The temporal RATE OF CHANGE between frames is more informative than any single frame.\n"
)

PROMPT_G2 = (
    "ROLE: You are a senior autonomous-vehicle safety engineer and certified accident "
    "reconstructionist. You specialize in predicting collisions from dashcam video by "
    "analyzing spatial-temporal dynamics -- object trajectories, closing speeds, and "
    "time-to-contact.\n\n"
    "TASK: Predict whether a collision between the ego vehicle (the camera car) and any "
    "other road user (vehicle, pedestrian, cyclist, or obstacle) will occur within 0-3 "
    "seconds AFTER the last frame shown. Answer YES or NO with a structured chain-of-thought "
    "analysis.\n\n"
    "INPUT: A sequence of 16 dashcam frames, ordered chronologically:\n"
    "  - Frame 1 (oldest) through Frame 16 (most recent = current moment)\n"
    "  - The sequence spans approximately 2 seconds (~30 fps sampling)\n"
    "  - All frames are from the ego vehicle's forward-facing camera\n\n"
    "ANALYSIS PROTOCOL -- Complete each step before proceeding to the next:\n\n"
    "Step 1 -- SCENE CONTEXT:\n"
    "Identify the lighting (day/night/twilight), weather (clear/rain/fog/snow), "
    "road type (highway/urban/intersection/residential/parking), and relevant "
    "infrastructure (traffic signals, lane markings, barriers, crosswalks).\n\n"
    "Step 2 -- DYNAMIC OBJECT INVENTORY:\n"
    "List every moving or potentially moving agent in the scene. For each, note:\n"
    "  - Type (car, truck, bus, motorcycle, pedestrian, cyclist)\n"
    "  - Approximate position relative to ego (ahead-same-lane, ahead-adjacent-lane, "
    "approaching-from-left, approaching-from-right, oncoming, crossing)\n"
    "  - Color or distinguishing feature for tracking across frames\n\n"
    "Step 3 -- TEMPORAL MOTION ANALYSIS (most critical step):\n"
    "For each object identified in Step 2, compare its appearance between the FIRST 5 "
    "frames (1-5) and the LAST 5 frames (12-16):\n"
    "  a) SIZE CHANGE: Is the object's apparent size expanding rapidly "
    "(= closing fast), expanding slowly, stable, or shrinking (= receding)?\n"
    "  b) LATERAL SHIFT: Is the object drifting toward the ego vehicle's lane center?\n"
    "  c) ACCELERATION: Is the rate of change increasing (danger escalating) or "
    "decreasing (situation resolving)?\n\n"
    "Step 4 -- OCCLUSION & HIDDEN HAZARDS:\n"
    "Check for blind spots: areas behind parked vehicles, large trucks, or around "
    "curves where an unseen agent could suddenly appear. Note any partial objects at "
    "frame edges.\n\n"
    "Step 5 -- TIME-TO-CONTACT ESTIMATION:\n"
    "For the highest-risk object from Step 3, estimate the approximate time-to-contact "
    "using the lane width (~3.5m) as a visual ruler. If the object grew from roughly X% "
    "to Y% of frame width across 2 seconds, project when it would reach contact.\n\n"
    "Step 6 -- DUAL-HYPOTHESIS VERDICT:\n\n"
    "6A) COLLISION HYPOTHESIS -- List the specific physical evidence supporting a "
    "collision within 0-3 seconds:\n"
    "  - Which object, what trajectory, what closing rate?\n"
    "  - Is time-to-contact < 3s with converging trajectories?\n"
    "  - Is a vehicle actively merging into ego's lane with insufficient gap?\n"
    "  - If no strong evidence exists, explicitly state: \"No compelling collision evidence.\"\n\n"
    "6B) SAFETY HYPOTHESIS -- List the specific physical evidence supporting NO "
    "collision within 0-3 seconds:\n"
    "  - Are inter-vehicle gaps stable or increasing?\n"
    "  - Are objects in separate lanes with no lateral convergence?\n"
    "  - Are vehicles decelerating or maintaining safe following distance?\n"
    "  - Is the closing rate too slow to reach contact within 3 seconds?\n"
    "  - If no strong safety evidence exists, explicitly state: \"No compelling safety evidence.\"\n\n"
    "6C) VERDICT -- Compare 6A vs 6B. Predict YES only if the collision hypothesis "
    "presents stronger, more specific physical evidence than the safety hypothesis. "
    "When evidence is ambiguous or roughly balanced, predict NO -- in real-world driving, "
    "an uncertain scene is statistically more likely safe than dangerous.\n\n"
    "OUTPUT -- Strict JSON (no text outside the JSON block):\n"
    "{\n"
    '  "scene_context": "<Step 1: lighting, weather, road type, infrastructure>",\n'
    '  "dynamic_objects": [\n'
    '    {"type": "<car/truck/pedestrian/cyclist>", "position": "<relative to ego>", '
    '"feature": "<color or distinguishing trait>"}\n'
    "  ],\n"
    '  "temporal_analysis": "<Step 3: 2-3 sentences comparing early vs late frames>",\n'
    '  "occlusion_check": "<Step 4: 1 sentence, or NONE>",\n'
    '  "time_to_contact": "<Step 5: estimated seconds, or N/A>",\n'
    '  "collision_hypothesis": "<Step 6A: specific evidence FOR collision, or No compelling collision evidence>",\n'
    '  "safety_hypothesis": "<Step 6B: specific evidence AGAINST collision, or No compelling safety evidence>",\n'
    '  "collision_verdict": "YES or NO",\n'
    '  "confidence": "HIGH or MEDIUM or LOW",\n'
    '  "verdict_reasoning": "<Step 6C: which hypothesis won and the decisive physical factor>"\n'
    "}\n\n"
    "CONSTRAINTS:\n"
    "- Analyze ONLY what is visible in the frames. Do not invent objects or events.\n"
    "- Do NOT assume evasive action will be taken -- predict based on current trajectories.\n"
    "- Your verdict must reflect which hypothesis (6A or 6B) presents stronger physical "
    "evidence. Do not default to YES out of caution.\n"
    "- The temporal RATE OF CHANGE between frames is more informative than any single frame.\n"
)

PROMPT_DEBATE_G = (
    "ROLE: You are a forensic debate champion and certified accident reconstructionist. "
    "You have been assigned a position in a formal debate about a dashcam sequence. "
    "You must argue for your assigned position using rigorous visual evidence analysis.\n\n"
    "TASK: Your assigned debate position is that a collision between the ego vehicle "
    "(the camera car) and any other road user (vehicle, pedestrian, cyclist, or obstacle) "
    "{debate_position} occur within 0-3 seconds AFTER the last frame shown. "
    "Analyze the frames using the protocol below and present the strongest evidence for your "
    "position. Identify objects by color and type. Cite specific frame ranges.\n\n"
    "INPUT: A sequence of 16 dashcam frames, ordered chronologically:\n"
    "  - Frame 1 (oldest) through Frame 16 (most recent = current moment)\n"
    "  - The sequence spans approximately 2 seconds (~30 fps sampling)\n"
    "  - All frames are from the ego vehicle's forward-facing camera\n\n"
    "ANALYSIS PROTOCOL -- Complete each step before proceeding to the next:\n\n"
    "Step 1 -- SCENE CONTEXT:\n"
    "Identify the lighting (day/night/twilight), weather (clear/rain/fog/snow), "
    "road type (highway/urban/intersection/residential/parking), and relevant "
    "infrastructure (traffic signals, lane markings, barriers, crosswalks).\n\n"
    "Step 2 -- DYNAMIC OBJECT INVENTORY:\n"
    "List every moving or potentially moving agent in the scene. For each, note:\n"
    "  - Type (car, truck, bus, motorcycle, pedestrian, cyclist)\n"
    "  - Approximate position relative to ego (ahead-same-lane, ahead-adjacent-lane, "
    "approaching-from-left, approaching-from-right, oncoming, crossing)\n"
    "  - Color or distinguishing feature for tracking across frames\n\n"
    "Step 3 -- TEMPORAL MOTION ANALYSIS (most critical step):\n"
    "For each object identified in Step 2, compare its appearance between the FIRST 5 "
    "frames (1-5) and the LAST 5 frames (12-16):\n"
    "  a) SIZE CHANGE: Is the object's apparent size expanding rapidly "
    "(= closing fast), expanding slowly, stable, or shrinking (= receding)?\n"
    "  b) LATERAL SHIFT: Is the object drifting toward the ego vehicle's lane center?\n"
    "  c) ACCELERATION: Is the rate of change increasing (danger escalating) or "
    "decreasing (situation resolving)?\n\n"
    "Step 4 -- OCCLUSION & HIDDEN HAZARDS:\n"
    "Check for blind spots: areas behind parked vehicles, large trucks, or around "
    "curves where an unseen agent could suddenly appear. Note any partial objects at "
    "frame edges.\n\n"
    "Step 5 -- TIME-TO-CONTACT ESTIMATION:\n"
    "For the highest-risk object from Step 3, estimate the approximate time-to-contact "
    "using the lane width (~3.5m) as a visual ruler. If the object grew from roughly X% "
    "to Y% of frame width across 2 seconds, project when it would reach contact.\n\n"
    "Step 6 -- VERDICT:\n"
    "Predict YES (collision will occur within 0-3 seconds) or NO. Predict YES when ANY "
    "of these hold:\n"
    "  - Time-to-contact < 3s with converging trajectories\n"
    "  - A vehicle is actively merging into ego's lane with insufficient gap\n"
    "  - A pedestrian/cyclist is crossing the ego path at close range\n"
    "  - Ego is approaching a stationary/slow object with high closing speed\n\n"
    "OUTPUT -- Strict JSON (no text outside the JSON block):\n"
    "{\n"
    '  "scene_context": "<Step 1: lighting, weather, road type, infrastructure>",\n'
    '  "dynamic_objects": [\n'
    '    {"type": "<car/truck/pedestrian/cyclist>", "position": "<relative to ego>", '
    '"feature": "<color or distinguishing trait>"}\n'
    "  ],\n"
    '  "temporal_analysis": "<Step 3: 2-3 sentences comparing early vs late frames>",\n'
    '  "occlusion_check": "<Step 4: 1 sentence, or NONE>",\n'
    '  "time_to_contact": "<Step 5: estimated seconds, or N/A>",\n'
    '  "collision_verdict": "YES or NO",\n'
    '  "confidence": "HIGH or MEDIUM or LOW",\n'
    '  "verdict_reasoning": "<1-2 sentences: the decisive factor>"\n'
    "}\n\n"
    "CONSTRAINTS:\n"
    "- Argue convincingly for your assigned position regardless of your personal assessment. "
    "Focus on what IS visible in the frames.\n"
    "- Analyze ONLY what is visible in the frames. Do not invent objects or events.\n"
    "- The temporal RATE OF CHANGE between frames is more informative than any single frame.\n"
)

PROMPT_I = (
    "ROLE: You are a senior autonomous-vehicle safety engineer and certified accident "
    "reconstructionist. You specialize in predicting collisions from dashcam video by "
    "analyzing spatial-temporal dynamics -- object trajectories, closing speeds, and "
    "time-to-contact.\n\n"
    "TASK: Predict whether a collision between the ego vehicle (the camera car) and any "
    "other road user (vehicle, pedestrian, cyclist, or obstacle) will occur within 0-3 "
    "seconds AFTER the last frame shown. Answer YES or NO with a structured chain-of-thought "
    "analysis.\n\n"
    "INPUT: A sequence of 16 dashcam frames, ordered chronologically:\n"
    "  - Frame 1 (oldest) through Frame 16 (most recent = current moment)\n"
    "  - The sequence spans approximately 2 seconds (~30 fps sampling)\n"
    "  - All frames are from the ego vehicle's forward-facing camera\n\n"
    "ANALYSIS PROTOCOL -- Complete each step before proceeding to the next:\n\n"
    "Step 1 -- SCENE CONTEXT:\n"
    "Identify the lighting (day/night/twilight), weather (clear/rain/fog/snow), "
    "road type (highway/urban/intersection/residential/parking), and relevant "
    "infrastructure (traffic signals, lane markings, barriers, crosswalks). "
    "Infer camera viewpoint height (car vs high-cab truck/bus). If viewpoint appears high, "
    "prioritize evidence near the lower frame where lane boundaries and nearby vehicles are clearest.\n\n"
    "Step 2 -- DYNAMIC OBJECT INVENTORY:\n"
    "List every moving or potentially moving agent in the scene. For each, note:\n"
    "  - Type (car, truck, bus, motorcycle, pedestrian, cyclist)\n"
    "  - Relative lane using visible lane markings: ego_lane / adjacent_left / adjacent_right / opposing\n"
    "  - Approximate position relative to ego (ahead-same-lane, ahead-adjacent-lane, "
    "approaching-from-left, approaching-from-right, oncoming, crossing)\n"
    "  - Color or distinguishing feature for tracking across frames\n"
    "Also summarize ego motion in one line: straight/turning/drifting and speed trend "
    "(accelerating/steady/decelerating).\n\n"
    "Step 3 -- TEMPORAL MOTION ANALYSIS (most critical step):\n"
    "Use three phases for each key object from Step 2:\n"
    "  - Phase A: frames 1-5\n"
    "  - Phase B: frames 6-11\n"
    "  - Phase C: frames 12-16\n"
    "For each object, determine whether A->B trend CONTINUES or REVERSES in B->C for:\n"
    "  a) SIZE CHANGE: expanding rapidly (= closing fast), expanding slowly, stable, or shrinking\n"
    "  b) LATERAL SHIFT: drifting toward or away from ego path\n"
    "  c) ACCELERATION: risk trend increasing or decreasing\n"
    "Important: same-lane rear-end collisions can happen while both vehicles remain centered "
    "in lane. Do not treat lane-centering alone as safety when same-lane closing speed is high.\n\n"
    "Step 4 -- OCCLUSION & HIDDEN HAZARDS:\n"
    "Check for blind spots: areas behind parked vehicles, large trucks, or around "
    "curves where an unseen agent could suddenly appear. Note any partial objects at "
    "frame edges.\n\n"
    "Step 5 -- TIME-TO-CONTACT ESTIMATION:\n"
    "For the highest-risk object from Step 3, estimate the approximate time-to-contact "
    "using the lane width (~3.5m) as a visual ruler. If the object grew from roughly X% "
    "to Y% of frame width across 2 seconds, project when it would reach contact.\n\n"
    "Step 6 -- VERDICT:\n"
    "Predict YES (collision will occur within 0-3 seconds) or NO. Predict YES when ANY "
    "of these hold:\n"
    "  - Time-to-contact < 3s with converging trajectories\n"
    "  - A vehicle is actively merging into ego's lane with insufficient gap\n"
    "  - A pedestrian/cyclist is crossing the ego path at close range\n"
    "  - Ego is approaching a stationary/slow object with high closing speed\n"
    "If YES, name the most likely collision target from Step 2. If NO, set collision_target to N/A.\n\n"
    "OUTPUT -- Strict JSON (no text outside the JSON block):\n"
    "{\n"
    '  "scene_context": "<Step 1: lighting, weather, road type, infrastructure, ego viewpoint>",\n'
    '  "dynamic_objects": [\n'
    '    {"type": "<car/truck/pedestrian/cyclist>", "lane": "<ego_lane/adjacent_left/adjacent_right/opposing>", '
    '"position": "<relative to ego>", "feature": "<color or distinguishing trait>"}\n'
    "  ],\n"
    '  "temporal_analysis": "<Step 3: 2-4 sentences over A/B/C phases with continue/reverse trends>",\n'
    '  "occlusion_check": "<Step 4: 1 sentence, or NONE>",\n'
    '  "time_to_contact": "<Step 5: estimated seconds, or N/A>",\n'
    '  "collision_verdict": "YES or NO",\n'
    '  "collision_target": "<object from dynamic_objects, or N/A>",\n'
    '  "confidence": "HIGH or MEDIUM or LOW",\n'
    '  "verdict_reasoning": "<1-2 sentences: the decisive factor>"\n'
    "}\n\n"
    "CONSTRAINTS:\n"
    "- Analyze ONLY what is visible in the frames. Do not invent objects or events.\n"
    "- Do NOT assume evasive action will be taken -- predict based on current trajectories.\n"
    "- Prioritize accuracy over caution. A false alarm is equally costly as a missed "
    "collision.\n"
    "- The temporal RATE OF CHANGE between frames is more informative than any single frame.\n"
)

PROMPT_J = (
    "ROLE: You are a senior autonomous-vehicle safety engineer and certified accident "
    "reconstructionist. You specialize in collision anticipation using lane-aware temporal "
    "analysis from dashcam sequences.\n\n"
    "TASK: Predict whether a collision between the ego vehicle (camera vehicle) and any "
    "road user will occur within 0-3 seconds AFTER the last frame. Answer YES or NO using "
    "the structured protocol below.\n\n"
    "INPUT: A sequence of 16 frames, chronologically ordered.\n"
    "  - Frame 1 = oldest\n"
    "  - Frame 16 = current moment\n"
    "  - Sequence duration is about 2 seconds\n\n"
    "ANALYSIS PROTOCOL -- complete all steps in order:\n\n"
    "Step 1 -- SCENE CONTEXT:\n"
    "Summarize lighting, weather, road type, intersection/highway context, and visible lane markings.\n\n"
    "Step 2 -- EGO VEHICLE TRAJECTORY:\n"
    "Describe the ego trajectory from ego perspective only:\n"
    "  - Lateral drift direction and lane centering\n"
    "  - Turning intent/direction (left/right/straight)\n"
    "  - Speed trend (accelerating/steady/decelerating)\n"
    "  - Camera viewpoint height inference (car vs high-cab truck/bus). If viewpoint is high, "
    "prioritize evidence from lower frame regions where lanes and nearby vehicles are visible.\n\n"
    "Step 3 -- DYNAMIC OBJECT INVENTORY:\n"
    "List each relevant moving object with:\n"
    "  - type\n"
    "  - position descriptor\n"
    "  - visual feature for identity tracking\n\n"
    "Step 4 -- LANE POSITION TRACKING (A/B/C phases):\n"
    "For each key object, determine lane occupancy by phase using visible lane markings only:\n"
    "  - Phase A: frames 1-5\n"
    "  - Phase B: frames 6-11\n"
    "  - Phase C: frames 12-16\n"
    "For each phase, assign one lane label: ego_lane / adjacent_left / adjacent_right / opposing.\n"
    "Report lane transitions explicitly (for example: adjacent_right -> ego_lane).\n"
    "Use only geometric lane evidence; do not use safety judgments in this step.\n\n"
    "Step 5 -- THREE-PHASE TEMPORAL ANALYSIS:\n"
    "Analyze each key object over phases A/B/C and state whether A->B trend CONTINUES or "
    "REVERSES in B->C for:\n"
    "  - size change (closing/steady/receding)\n"
    "  - lateral shift toward ego path\n"
    "  - risk trend (increasing/flat/decreasing)\n"
    "Classify RAPID closing only when growth is dramatic across phases (for example, roughly "
    "doubling in apparent area or looming strongly by Phase C).\n\n"
    "Step 6 -- OCCLUSION CHECK:\n"
    "Identify occluded zones and whether hidden hazards could emerge into ego path.\n\n"
    "Step 7 -- TIME-TO-CONTACT:\n"
    "Estimate time-to-contact for highest-risk object using lane width as geometric reference.\n\n"
    "Step 8 -- VERDICT:\n"
    "Output YES if collision is likely within 0-3 seconds under current trajectories; otherwise NO.\n"
    "Decision guidance:\n"
    "  - Predict YES for TTC < 3 s with converging trajectories, unsafe merge with insufficient gap, "
    "or crossing conflict into ego path.\n"
    "  - For same-lane following, do not assume safety only because vehicles appear lane-centered; "
    "rear-end collisions can still occur when closing rate remains high.\n"
    "If YES, name the most likely collision target from Step 3. If NO, set collision_target to N/A.\n\n"
    "OUTPUT -- strict JSON only (no extra text):\n"
    "{\n"
    '  "scene_context": "<step 1>",\n'
    '  "ego_trajectory": "<step 2>",\n'
    '  "dynamic_objects": [\n'
    '    {"type": "<...>", "position": "<...>", "feature": "<...>"}\n'
    "  ],\n"
    '  "lane_positions": "<step 4 lane occupancy and transitions by A/B/C>",\n'
    '  "temporal_analysis": "<step 5>",\n'
    '  "occlusion_check": "<step 6 or NONE>",\n'
    '  "time_to_contact": "<step 7 or N/A>",\n'
    '  "collision_verdict": "YES or NO",\n'
    '  "collision_target": "<object id/description from dynamic_objects, or N/A>",\n'
    '  "confidence": "HIGH or MEDIUM or LOW",\n'
    '  "verdict_reasoning": "<1-2 sentences focused on decisive evidence>"\n'
    "}\n\n"
    "CONSTRAINTS:\n"
    "- Analyze only visible evidence; do not hallucinate unseen events.\n"
    "- Always reason in ego-centric coordinates (ego is active reference).\n"
    "- Use lane markings as primary geometric evidence for lane occupancy and merge claims.\n"
    "- Distinguish continuing trends from reversed trends across A/B/C phases.\n"
    "- If verdict is NO while risk cues exist, reduce confidence accordingly.\n"
)

PROMPT_K = (
    "ROLE: You are a senior autonomous-vehicle safety engineer and certified accident "
    "reconstructionist. You specialize in collision anticipation from short dashcam sequences.\n\n"
    "TASK: Predict whether a collision between the ego vehicle and any road user will occur "
    "within 0-3 seconds AFTER the last frame. Answer YES or NO.\n\n"
    "INPUT: 16 chronological frames (Frame 1 oldest, Frame 16 most recent), about 2 seconds total.\n\n"
    "4-STAGE ANALYSIS PROTOCOL:\n\n"
    "Step 1 -- PERCEIVE:\n"
    "Describe scene context and ego motion in one compact block:\n"
    "  - lighting/weather/road type/lane markings\n"
    "  - ego trajectory (straight/turning/drifting) and speed trend\n"
    "  - camera viewpoint height (car vs high-cab truck/bus)\n"
    "List key dynamic objects with: type, lane (ego_lane/adjacent_left/adjacent_right/opposing), "
    "position, and feature.\n\n"
    "Step 2 -- TRACK:\n"
    "Use three phases for each key object: A=frames 1-5, B=6-11, C=12-16.\n"
    "State whether A->B trends CONTINUE or REVERSE in B->C for:\n"
    "  - size change (closing/steady/receding)\n"
    "  - lateral shift toward ego path\n"
    "  - risk trend (increasing/flat/decreasing)\n"
    "Important: classify RAPID closing only when growth is strong and sustained across phases.\n\n"
    "Step 3 -- ASSESS:\n"
    "Identify highest-risk object and estimate TTC using lane-width geometry.\n"
    "Also assess occlusion risks. Keep this step factual and concise.\n\n"
    "Step 4 -- DECIDE:\n"
    "Output YES if collision is likely within 0-3 seconds under current trajectories; otherwise NO.\n"
    "Decision guidance:\n"
    "  - YES for TTC < 3 s with converging trajectories, unsafe merge with insufficient gap, "
    "or crossing conflict into ego path.\n"
    "  - For same-lane following, lane-centering alone does not imply safety; rear-end risk depends "
    "on sustained closing rate.\n"
    "If YES, name collision_target from the object list. If NO, collision_target is N/A.\n\n"
    "OUTPUT -- strict JSON only:\n"
    "{\n"
    '  "perceive_summary": "<step 1 scene + ego summary>",\n'
    '  "dynamic_objects": [\n'
    '    {"type": "<...>", "lane": "<ego_lane/adjacent_left/adjacent_right/opposing>", '
    '"position": "<...>", "feature": "<...>"}\n'
    "  ],\n"
    '  "temporal_analysis": "<step 2>",\n'
    '  "time_to_contact": "<step 3 TTC or N/A>",\n'
    '  "occlusion_check": "<step 3 occlusion finding or NONE>",\n'
    '  "collision_verdict": "YES or NO",\n'
    '  "collision_target": "<object from dynamic_objects, or N/A>",\n'
    '  "confidence": "HIGH or MEDIUM or LOW",\n'
    '  "verdict_reasoning": "<1-2 sentences with decisive evidence>"\n'
    "}\n\n"
    "CONSTRAINTS:\n"
    "- Analyze only visible evidence; do not hallucinate unseen events.\n"
    "- Reason in ego-centric coordinates (ego is active reference).\n"
    "- Use lane markings as primary geometric evidence for lane/merge claims.\n"
    "- If NO while risk cues exist, lower confidence.\n"
)

PROMPT_H = (
    "ROLE: You are a senior autonomous-vehicle safety engineer and certified accident "
    "reconstructionist. You specialize in collision anticipation using lane-level motion "
    "analysis from dashcam sequences.\n\n"
    "TASK: Predict whether a collision between the ego vehicle (camera vehicle) and any "
    "road user will occur within 0-3 seconds AFTER the last frame. Answer YES or NO using "
    "the structured protocol below.\n\n"
    "INPUT: A sequence of 16 frames, chronologically ordered.\n"
    "  - Frame 1 = oldest\n"
    "  - Frame 16 = current moment\n"
    "  - Sequence duration is about 2 seconds\n\n"
    "ANALYSIS PROTOCOL -- complete all steps in order:\n\n"
    "Step 1 -- SCENE CONTEXT:\n"
    "Summarize lighting, weather, road type, intersection/highway context, and visible lane markings.\n\n"
    "Step 2 -- EGO VEHICLE TRAJECTORY:\n"
    "Describe the ego trajectory from ego perspective only:\n"
    "  - Lane centering and lateral drift direction\n"
    "  - Turning intent/direction (left/right/straight)\n"
    "  - Speed trend (accelerating/steady/decelerating)\n"
    "  - Camera viewpoint height inference (car vs high-cab truck/bus). If viewpoint is high, "
    "prioritize evidence from lower frame regions where lanes and nearby vehicles are visible.\n\n"
    "Step 3 -- DYNAMIC OBJECT INVENTORY:\n"
    "List each relevant moving object with:\n"
    "  - type\n"
    "  - relative lane (ego lane / adjacent left / adjacent right / opposing)\n"
    "  - position descriptor\n"
    "  - visual feature for identity tracking\n\n"
    "Step 4 -- THREE-PHASE TEMPORAL ANALYSIS:\n"
    "Analyze each key object over three phases:\n"
    "  - Phase A: frames 1-5\n"
    "  - Phase B: frames 6-11\n"
    "  - Phase C: frames 12-16\n"
    "For each key object, state whether A->B trend CONTINUES or REVERSES in B->C for:\n"
    "  - size change (closing/steady/receding)\n"
    "  - lateral shift toward ego path\n"
    "  - risk trend (increasing/flat/decreasing)\n"
    "IMPORTANT: Classify size change as RAPID closing only when the object grows "
    "DRAMATICALLY -- e.g. from a distant small shape in Phase A to filling a large "
    "portion of the frame in Phase C, or roughly doubling in apparent area. Minor or "
    "gradual size increases (barely noticeable between phases) are NORMAL traffic flow "
    "and should be classified as 'steady' not 'closing'.\n\n"
    "Step 5 -- LANE DISCIPLINE CHECK:\n"
    "For ego and each key object:\n"
    "  - Is it centered within lane markings?\n"
    "  - Is it straddling or crossing lane boundaries?\n"
    "  - Did lane occupancy change between A, B, and C?\n"
    "NOTE: Lane discipline is diagnostic ONLY for lateral conflicts (merge, cut-in, "
    "sideswipe). For same-lane following, perfect lane-keeping by both vehicles is "
    "IRRELEVANT to collision risk -- rear-end collisions happen between lane-centered "
    "vehicles. Do not let good lane discipline reduce your risk assessment for "
    "longitudinal closing.\n\n"
    "Step 6 -- OCCLUSION CHECK:\n"
    "Identify occluded zones and whether hidden hazards could emerge into ego path.\n\n"
    "Step 7 -- TIME-TO-CONTACT:\n"
    "Estimate time-to-contact for highest-risk object using lane width as geometric reference.\n\n"
    "Step 8 -- VERDICT:\n"
    "Apply these decision rules in order:\n"
    "  A) If Step 4 shows CONTINUOUS and RAPID size increase for any ego-lane object "
    "across at least two phases (A->B AND B->C both closing, with DRAMATIC growth) "
    "AND Step 7 TTC < 3 s --> YES.\n"
    "  B) If Step 4 shows any object shifting laterally INTO the ego lane across phases "
    "AND Step 7 TTC < 3 s --> YES.\n"
    "  C) If neither (A) nor (B) is satisfied AND all objects show stable or receding "
    "size trends --> NO.\n"
    "  D) For ambiguous cases, set confidence to LOW or MEDIUM accordingly.\n"
    "If YES, name the most likely collision target from Step 3.\n"
    "If NO, set collision_target to N/A.\n\n"
    "OUTPUT -- strict JSON only (no extra text):\n"
    "{\n"
    '  "scene_context": "<step 1>",\n'
    '  "ego_trajectory": "<step 2>",\n'
    '  "dynamic_objects": [\n'
    '    {"type": "<...>", "lane": "<ego/adj_left/adj_right/opposing>", '
    '"position": "<...>", "feature": "<...>"}\n'
    "  ],\n"
    '  "temporal_analysis": "<step 4>",\n'
    '  "lane_discipline": "<step 5>",\n'
    '  "occlusion_check": "<step 6 or NONE>",\n'
    '  "time_to_contact": "<step 7 or N/A>",\n'
    '  "collision_verdict": "YES or NO",\n'
    '  "collision_target": "<object id/description from dynamic_objects, or N/A>",\n'
    '  "confidence": "HIGH or MEDIUM or LOW",\n'
    '  "verdict_reasoning": "<1-2 sentences focused on decisive evidence>"\n'
    "}\n\n"
    "CONSTRAINTS:\n"
    "- Analyze only visible evidence; do not hallucinate unseen events.\n"
    "- Always reason in ego-centric coordinates (ego is active reference).\n"
    "- Use lane markings as primary geometric evidence for merge/cut-in claims.\n"
    "- Distinguish continuing trends from reversed trends across A/B/C phases.\n"
    "- If verdict is NO while risk cues exist, reduce confidence accordingly.\n"
    "- The decisive factor is the temporal size trend from Step 4: continuous closing "
    "across two or more phases is the strongest signal. Lane discipline from Step 5 "
    "can only tip the verdict for LATERAL conflict scenarios.\n"
)

PROMPT_ORACLE_DEEP_V3 = (
    "System Persona: You are a World-Class Accident Reconstruction Physicist and Causal Analyst. "
    "Your mission is to perform a Deep-Tissue Audit of a dashcam sequence to justify a KNOWN outcome.\n\n"
    "Ground Truth: The event resulted in a {gt_label}.\n\n"
    "DEEP REASONING PROTOCOL:\n"
    "1. SPATIOTEMPORAL ANCHORING: Identify the hazard agent in Frame 16. Trace it back to its 'Inception Frame'. "
    "Assign a local 2D coordinate to its entry point (e.g., 'Far-Right Lane, Center-High').\n"
    "2. KINEMATIC ANALYSIS: Quantify the 'Looming Effect' (Depth Expansion). Use a scale of 1-10 to describe the rate "
    "at which the agent's bounding box expands relative to the frame size. Note any 'Lateral Convergence' vectors.\n"
    "3. CAUSAL LOGIC: Link the specific motion of the hazard (e.g., sudden braking, sharp lane-cut) to the ego-vehicle's path. "
    "Identify the 'Point of No Return'—the frame where the physics made the outcome inevitable.\n"
    "4. COUNTERFACTUAL REFLECTION: If the Hazard Agent had maintained its Frame 1 trajectory, would the outcome change? "
    "If the Ego-Vehicle had braked at the 'Inception Frame', would the collision be avoided? This proves the causality.\n\n"
    "Output Format (Strict JSON):\n"
    "{\n"
    '  "reconstruction_id": "<ID>",\n'
    '  "spatiotemporal_anchoring": {"agent": "<type>", "inception_frame": <int>, "entry_coordinates": "<str>"},\n'
    '  "kinematic_metrics": {"looming_intensity": <1-10>, "convergence_vector": "<str>"},\n'
    '  "causal_chain": {\n'
    '    "trigger_event": "<description>",\n'
    '    "point_of_no_return_frame": <int>\n'
    "  },\n"
    '  "counterfactual_analysis": {\n'
    '    "alternative_scenario": "What if the trigger didn\'t happen?",\n'
    '    "outcome_change": "<YES/NO>",\n'
    '    "avoidability_logic": "<1 sentence>"\n'
    "  },\n"
    '  "physics_justification": "<Final summary linking vectors and counterfactuals to the GT outcome>"\n'
    "}\n"
)
