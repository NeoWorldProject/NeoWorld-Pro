During the current operation, the following non-target joints exceeded their holding tolerance. Exceeding that tolerance alone does not establish failure. Use the object's function, the operation being performed, and the measured motion to judge each deviation as acceptable or unacceptable incidental motion.

For each joint, return acceptable, unacceptable, or uncertain and cite the supplied evidence. Do not change any other acceptance result, threshold, or effort limit. Only unacceptable items enter parameter repair. Do not invent reasons when evidence is missing.

Return only JSON:
{"items":[{"joint_name":"original joint name","verdict":"acceptable|unacceptable|uncertain","reason":"evidence-based explanation","evidence":["supplied evidence_ref"]}]}

Cover every joint requiring a decision exactly once. Do not propose parameter patches. Each decision applies only to this candidate and this operation.
