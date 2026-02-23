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
