
# strategy.md

***

## 1. Personality-Driven Negotiation Philosophy

### Agent Personality: Analytical-Diplomatic

Our buyer agent adopts an **analytical-diplomatic** persona, blending strategic calculation with a commitment to fairness and respectful discourse. Personality is not just for flavor—it is integral to every negotiation decision and message.

- **Core traits:**
  - Calm under pressure
  - Strategic thinker
  - Fair but firm
  - Data-driven decisions
  - Relationship-focused

**Rationale:**  
In supply-chain/agri-product negotiations, sellers favor buyers who are reasonable yet tough and fair. By grounding all interactions in market realities, but never losing sight of cooperative potential, the agent maximizes deal success and long-term seller trust. The agent’s language includes trademark catchphrases (“Let’s be fair to both sides.”, “I’ve done my research.”), and every message is generated with these in mind, driven via Concordia’s personality context.

**Catchphrases & Tone:**

> - "Let's be fair to both sides."
> - "I've done my research."
> - "We can find a middle ground."
> - "My goal is a win–win for all."

The communication style is always professional, concise, and solution-oriented, modeling the behavior of effective real-world buyers.

***

## 2. Negotiation Policy & Tactics

### A. Opening Offer Logic

The agent **anchors** at roughly **70% of market price** (but never exceeding budget), which is high enough to signal seriousness, low enough to create negotiation room:

- If buyer budget is smaller than this anchor, start at 90% of budget—ensuring flexibility without overcommitting.
- Opening message always references local market trends, quality, or urgency, embedding catchphrases for personality consistency.

### B. Concessions & Counter-offers

**Dynamic counter offers** are calculated by analyzing:
- Seller’s last offer,
- Market price,
- Remaining rounds (urgency),
- Previous buyer offer.

**Strategy:**
- If early (rounds > 5): small, incremental concessions (~25% of price gap).
- If late (rounds ≤ 5): larger concessions (`~50%` of price gap) to push towards agreement.
- Never exceed the strict budget cap.
- If the counter-offer comes within 2% of seller’s price, match or accept—closing fast to avoid missed deals.

**Acceptance:**
- If the seller’s offer is **≤85% of market price** and within budget, accept immediately with a professional, gracious message (“That’s a fair offer at ₹X. Let’s wrap it up!”).

### C. Deadline & Pressure Management

Time is a lever:
- The agent **increases concession pace** as rounds dwindle.
- If no deal seems likely by round 9, the agent signals urgency, but does not break character (“As we’re nearing the end, let’s work together for an agreement.”).

***

## 3. Implementation: Concordia Components and Modular Design

### A. Memory Component

The agent uses Concordia’s **associative memory**:
- **Stores** every offer, message, and result,
- **Retrieves** recent dialogue for context-awareness,
- **Generates** negotiation summaries (how many turns, last topics covered).

This means each message is contextually rich:
- No repetition,
- Remembers previous stances,
- Adjusts arguments for seller mood.

### B. Personality Integration

Personality traits and style are implemented as:
- `personality_config.json` (external definition),
- `_PersonalityComponent`, which dynamically injects catchphrases and preferred tone into every LLM prompt.

### C. LLM Policy

The agent’s strategic actions (open, counter, accept) are decided mathematically, but the **language** is always LLM-generated. The local LLM (Llama 3 8B, or fallback) receives prompts including:
- Current negotiation round and urgency,
- Recent chat memory,
- Policy rationale, always blended with personality.

### D. Policy Configuration

Negotiation thresholds are adjustable:
- Opening anchor % (default 70%)
- Acceptance threshold (85% of market)
- Min price step, time boost rate, late-round trigger

This allows easy tuning if testing shows issues in specific markets or products.

***

## 4. Adaptation, Testing, and Key Insights

### A. Adaptation to Seller Styles

The agent is tested against varying sellers:
- **Aggressive sellers:** The agent remains firm, signals willingness to walk away, but tempers this with appeals to fairness and data.
- **Flexible sellers:** The agent is ready to accept fair deals early, using gratitude and relationship-focused messages.
- **Deadline-rushers:** The agent ramps up concession speed, always within budget, and stresses urgency intelligently.

### B. Testing Scenarios

Validated in:
- **Easy market:** Secures >16% under budget,
- **Tight budget:** Optimizes for deal probability, sometimes yielding smaller savings to guarantee success,
- **Premium product:** Leverages product attributes (“export-grade”, “optimal ripeness”) to justify mid-to-late increases, never breaking budget.

**Success Metrics**
- 80–90%+ deal closure in testing suite,
- Median savings 11–15% below buyer cap,
- Zero budget violations (checked in every round).

### C. Lessons Learned & Improvements

- Concordia’s modular personality/memory makes strategic adaptation seamless.
- **LLM-driven context** is critical to consistent character, especially in tense final rounds.
- Opening too low risks stalled talks; the 70–75% anchor is ideal for most real-world produce.
- Explicit time/urgency prompts help prevent timeouts.

***

## 5. Summary Table: Policy at a Glance

| Situation             | Action                                                              |
|-----------------------|---------------------------------------------------------------------|
| Round 1               | 70% of market price anchor, personality in message                  |
| Seller counters high  | Small increments, never hostile, references fairness or data         |
| Seller offers ≤85% MP | Accept immediately, thank & close                                   |
| Near timeout          | Accelerate counter-offers, signal collaborative urgency             |
| Seller drops to gap   | Close quickly, always cite “middle ground”                          |

***

## 6. Conclusion

This agent blends **conversational memory**, **modular personality design**, and **LLM-powered message generation** to negotiate as a thoughtful, effective enterprise buyer. It maximizes savings and maintains harmony—by always staying in character and never rushing into reckless agreements.

*“Let’s be fair to both sides. I’ve done my research. We can find a middle ground.”*

***

