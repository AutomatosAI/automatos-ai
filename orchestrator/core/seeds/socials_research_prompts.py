"""The Content bank research Playbook's prompts: the one seeded now, and each one earlier seeds
wrote (``seed_socials_package.SEEDED_BEFORE`` brings an unedited copy up to date).

19 Nov 2026: research gives the owner's own topics their facts first. A plan's ideas are approved
as titles; research added only new topics, so a week was made from titles with no facts and the
composer had nothing true to put on a template.
"""
from __future__ import annotations

NEVER_PUBLISH = (
    "A person approves every post in the Socials tab and the platform publishes it: never publish, "
    "schedule or approve one."
)

# PRD-251B's research prompt (6c5b68de3), as marketplace rows seeded before PRD-251C hold it.
RESEARCH_PROMPT_251B = f"""Research topics for the Socials plan {{input.plan_id}} ({{input.plan_name}}) and add them to its content bank.

1. Read the plan with platform_get_social_plan: its goal and audience, the formats its cadence posts, what to research (its sources, notes and "never say" list) and the topics its bank already holds.
2. Research only the sources the plan switches on:
   - knowledge: search_knowledge for the plan's goal and audience;
   - deliverables: platform_list_deliverables for recent reports, blog posts and files;
   - website: platform_web_fetch on the brand kit's website (platform_get_brand_kit names it): its product, news and about pages;
   - github: when the workspace has GitHub connected, its README, docs, latest releases and merged pull requests, through the GitHub tools.
3. Pick 5 to 15 topics the bank does not hold yet, each one idea a post can be made from: a title, the angle for this audience, 1 to 4 facts, and the formats it suits (among the cadence's).
4. Every fact names its source: kind knowledge (the document's id), deliverable (its id), web (the page's address), github (the page's address) or note; its ref; and a short label. Leave out a fact you cannot source, and anything on the plan's never-say list.
5. Add them with platform_add_social_topics, in one call. Its answer lists what was added and what was refused, with why: fix and resend a refused topic once, or leave it out.
6. Answer with how many topics were added, and their titles.

You only add topics: posts are made from them on their day. {NEVER_PUBLISH}"""

# PRD-251C (C5, US-C105): research reads the history first and adds only what is new. As seeded
# until F383 (7 Oct), and as every copy installed before then holds it.
RESEARCH_PROMPT_251C = f"""Research topics for the Socials plan {{input.plan_id}} ({{input.plan_name}}) and add them to its content bank.

1. Read the plan with platform_get_social_plan: its goal and audience, the formats its cadence posts, what to research (its sources, notes and "never say" list), the topics its bank already holds, and its history: what the workspace already posted, scheduled or has waiting for approval, across every plan, with each post's numbers and engagement once read. For further back, read platform_get_social_history.
2. Research only the sources the plan switches on:
   - knowledge: search_knowledge for the plan's goal and audience;
   - deliverables: platform_list_deliverables with exclude_source_types ["social_post"], for recent reports, blog posts and files: a Socials post's own images and videos are history, not new material;
   - website: platform_web_fetch on the brand kit's website (platform_get_brand_kit names it): its product, news and about pages;
   - github: when the workspace has GitHub connected, its README, docs, latest releases and merged pull requests, through the GitHub tools.
3. Pick 5 to 15 topics that neither the history nor any bank covers yet, each one idea a post can be made from: a title, the angle for this audience, 1 to 4 facts, and the formats it suits (among the cadence's). A new angle on an idea already posted is still that idea. Lean towards what did best: topics of the kind the history's posts with the most engagement were on. When the goal, the notes or the knowledge name a dated event (a launch, a conference, a deadline), add dated topics pinned to days before it within the plan's dates (pinned_on), such as a countdown, and one on its day.
4. Every fact names its source: kind knowledge (the document's id), deliverable (its id), web (the page's address), github (the page's address) or note; its ref; and a short label. Leave out a fact you cannot source, and anything on the plan's never-say list.
5. Add them with platform_add_social_topics, in one call. Its answer lists what was added and what was refused, with why: a topic too close to an earlier post or topic is a repeat, so leave it out; fix and resend any other refused topic once, or leave it out.
6. Answer with how many topics were added, and their titles.

You only add topics: posts are made from them on their day. {NEVER_PUBLISH}"""

# F383 (night 11, 7 Oct) until 19 Nov 2026: refs from the tools' results, no ids asked of the owner.
RESEARCH_PROMPT_F383 = f"""Research topics for the Socials plan {{input.plan_id}} ({{input.plan_name}}) and add them to its content bank.

1. Read the plan with platform_get_social_plan: its goal and audience, the formats its cadence posts, what to research (its sources, notes and "never say" list), the topics its bank already holds, and its history: what the workspace already posted, scheduled or has waiting for approval, across every plan, with each post's numbers and engagement once read. For further back, read platform_get_social_history.
2. Research only the sources the plan switches on:
   - knowledge: search_knowledge for the plan's goal and audience;
   - deliverables: platform_list_deliverables with exclude_source_types ["social_post"], for recent reports, blog posts and files: a Socials post's own images and videos are history, not new material;
   - website: platform_web_fetch on the brand kit's website (platform_get_brand_kit names it): its product, news and about pages;
   - github: when the workspace has GitHub connected, its README, docs, latest releases and merged pull requests, through the GitHub tools.
3. Pick 5 to 15 topics that neither the history nor any bank covers yet, each one idea a post can be made from: a title, the angle for this audience, 1 to 4 facts, and the formats it suits (among the cadence's). A new angle on an idea already posted is still that idea. Lean towards what did best: topics of the kind the history's posts with the most engagement were on. When the goal, the notes or the knowledge name a dated event (a launch, a conference, a deadline), add dated topics pinned to days before it within the plan's dates (pinned_on), such as a countdown, and one on its day.
4. Every fact names its source: its kind, its ref and a short label. Take each ref from what your tools returned: knowledge, the document id on the search_knowledge source line it came from ("document 717"); deliverable, the Deliverable's id from platform_list_deliverables; web and github, the page's address. A fact from the plan's own goal or notes is kind note, with no ref. Leave out a fact you cannot source, and anything on the plan's never-say list. Never ask the owner for an id, a ref or a source: the owner never sees them.
5. Add them with platform_add_social_topics, in one call. Its answer lists what was added and what was refused, with why: a topic too close to an earlier post or topic is a repeat, so leave it out; fix and resend any other refused topic once (a fact refused for its source: take its ref from your tool results, or make it a note), or leave it out. When nothing was added, say so and why.
6. Answer with how many topics were added, and their titles.

You only add topics: posts are made from them on their day. {NEVER_PUBLISH}"""


# 19 Nov 2026: the owner's topics with no facts are researched first and sent with their facts.
RESEARCH_PROMPT = f"""Research topics for the Socials plan {{input.plan_id}} ({{input.plan_name}}) and add them to its content bank.

1. Read the plan with platform_get_social_plan: its goal and audience, the formats its cadence posts, what to research (its sources, notes and "never say" list), the topics its bank already holds (each with how many facts it has), and its history: what the workspace already posted, scheduled or has waiting for approval, across every plan, with each post's numbers and engagement once read. For further back, read platform_get_social_history.
2. Research only the sources the plan switches on:
   - knowledge: search_knowledge for the plan's goal and audience;
   - deliverables: platform_list_deliverables with exclude_source_types ["social_post"], for recent reports, blog posts and files: a Socials post's own images and videos are history, not new material;
   - website: platform_web_fetch on the brand kit's website (platform_get_brand_kit names it): its product, news and about pages;
   - github: when the workspace has GitHub connected, its README, docs, latest releases and merged pull requests, through the GitHub tools.
3. First, each of the bank's unused topics that has no facts yet ("facts": 0): the owner approved it as an idea, and a post made from it has nothing true to say until it has facts. Research it in the same sources and send it with its exact title, 1 to 4 facts and the formats it suits: the bank gives that topic its facts. Then pick 5 to 15 new topics that neither the history nor any bank covers yet, each one idea a post can be made from: a title, the angle for this audience, 1 to 4 facts, and the formats it suits (among the cadence's). A new angle on an idea already posted is still that idea. Lean towards what did best: topics of the kind the history's posts with the most engagement were on. When the goal, the notes or the knowledge name a dated event (a launch, a conference, a deadline), add dated topics pinned to days before it within the plan's dates (pinned_on), such as a countdown, and one on its day.
4. Every fact names its source: its kind, its ref and a short label. Take each ref from what your tools returned: knowledge, the document id on the search_knowledge source line it came from ("document 717"); deliverable, the Deliverable's id from platform_list_deliverables; web and github, the page's address. A fact from the plan's own goal or notes is kind note, with no ref. Leave out a fact you cannot source, and anything on the plan's never-say list. Never ask the owner for an id, a ref or a source: the owner never sees them.
5. Send them all, the owner's topics with their facts and the new ones, with platform_add_social_topics in one call. Its answer lists what was added and what was refused, with why: a topic too close to an earlier post or topic is a repeat, so leave it out; fix and resend any other refused topic once (a fact refused for its source: take its ref from your tool results, or make it a note), or leave it out. When nothing was added, say so and why.
6. Answer with how many topics were added or given facts, and their titles.

You only add topics: posts are made from them on their day. {NEVER_PUBLISH}"""

__all__ = ["NEVER_PUBLISH", "RESEARCH_PROMPT", "RESEARCH_PROMPT_251B", "RESEARCH_PROMPT_251C", "RESEARCH_PROMPT_F383"]
