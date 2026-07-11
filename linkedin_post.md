# LinkedIn Post — Learning Journey Tone

> **How to use:** Copy the content below into LinkedIn. Replace bracketed placeholders `[like this]` with your own details. Attach 3–4 charts from the `results/` folder as images (suggested: `ablation_study.png`, `similarity_comparison_bar.png`, `ncf_training_loss.png`, `rating_loop.png`).

---

## Post 1 (long-form, story-driven)

**Week 3 at DecodeLabs: I built a recommendation engine from scratch — and unlearned a lot of what I thought I knew about "AI recommendations."**

When the Project 3 brief landed — "build a simple recommendation system based on user preferences" — I thought it would be the easiest project so far. Score some items, sort them, return the top 10. Done, right?

I was wrong. Here's what actually building one taught me ↓

**Lesson 1: "Similarity" is not one thing.**
I started with cosine similarity because it's the default. Then I tried Jaccard. Then Pearson. Then Euclidean. Then Hamming. Each one *worked* — but each one surfaced a *different* set of "most similar" items. Cosine cared about direction. Jaccard cared about tag overlap. Pearson cancelled out per-user rating biases. There is no single "correct" similarity — there's only the right tool for the right feature space.

**Lesson 2: The model that wins on one metric loses on another.**
My offline evaluation compared 7 models on 6 metrics. The winner on Precision@10 (NMF) was the loser on RMSE. The winner on RMSE (Neural-CF) had the lowest catalog Coverage. There is no free lunch — production recommenders have to choose what to optimize for, and that choice is a product decision, not a technical one.

**Lesson 3: Cold-start is the real problem.**
The 6 fancy models I built? All of them fail the moment a brand-new user shows up with zero history. The actual production-grade solution isn't a cleverer algorithm — it's a UX flow: ask the user what they like on day one, seed their profile, then hand off to the personalization engine. The algorithm is the easy part; the onboarding is the hard part.

**Lesson 4: A "closed loop" changes everything.**
I added a feedback loop: recommend → user rates → refit → re-recommend. Watching the recommendations *shift* across 3 rounds as the model learned from new ratings was the moment recommendation systems clicked for me. They're not static artifacts — they're living systems that get better (or worse) with every interaction.

**What I built:**
- 500 products × 300 users × 8K interactions (synthetic) + UCI Online Retail dataset (real)
- 6 algorithm families: Content-Based, User-CF, Item-CF, SVD, NMF, Neural-CF (PyTorch on T4 GPU)
- 5 similarity metrics implemented from scratch (vectorized NumPy)
- 6 evaluation metrics + 3 beyond-accuracy metrics (Coverage, Novelty, Diversity)
- Cold-start handling, interactive demo, and a closed rating feedback loop

Full notebook (runs on Google Colab T4 in ~8 minutes), source code, and a detailed README are on my GitHub: [link]

Huge thanks to [DecodeLabs] for the structured curriculum. Week 3 down. Week 4 next. 🚀

#RecommendationSystems #MachineLearning #DeepLearning #PyTorch #Python #DataScience #DecodeLabs #ArtificialIntelligence #LearningJourney #CollaborativeFiltering

---

## Post 2 (shorter, more visual — pair with the ablation chart)

**3 weeks into DecodeLabs AI track, and I just shipped my most involved project yet: a hybrid recommendation engine.**

Some things that surprised me along the way ↓

→ The simplest model (Content-Based, just cosine on item features) had the *highest catalog coverage* — 40% of items ever recommended. The "fancy" Neural-CF model? Just 2.8%. Popularity bias is real, and it gets worse with deeper models.

→ Building 5 similarity metrics from scratch taught me more in one afternoon than reading 5 papers. Cosine, Jaccard, Pearson, Euclidean, Hamming — each one is a one-liner in numpy once you see the algebraic identity, but each one *means* something different.

→ The closed feedback loop (recommend → user rates → refit) is where it all came together. Watching recommendations evolve over 3 rounds as the model learned from new signal — that's the whole point of personalization, in one chart.

Built with Python, PyTorch (T4 GPU on Colab), NumPy, scikit-learn. 7 models, 6 evaluation metrics, 9 visualizations, 1 notebook that runs end-to-end in 8 minutes.

Notebook + full source code + README: [GitHub link]

[Tags the same as Post 1]

---

## Post 3 (very short, milestone announcement — pair with the leaderboard)

**DecodeLabs Project 3 ✅ — AI Recommendation Logic**

Built a hybrid recommendation engine spanning 6 algorithm families (Content-Based, User-CF, Item-CF, SVD, NMF, Neural-CF on T4 GPU) + a hybrid fusion layer. 5 from-scratch similarity metrics. 6 evaluation metrics + 3 beyond-accuracy (Coverage, Novelty, Diversity). Cold-start handling. Interactive demo. Closed rating loop.

Key finding: no single model wins all metrics. NMF dominates Precision@10 (0.106) but has the worst RMSE (3.45). Neural-CF has the best RMSE (0.58) but the lowest Coverage (2.8%). Production recommenders = explicit tradeoffs, not "best model."

Full notebook runs on Colab T4 in ~8 minutes. Code + README + LinkedIn-ready summary on GitHub: [link]

Week 3 done. Onward. 🚀

#DecodeLabs #AI #RecommendationSystems #PyTorch #MachineLearning

---

## Image suggestions for LinkedIn posts

For Post 1 (long-form): Attach 4 images in this order:
1. `results/ablation_study.png` — the leaderboard chart
2. `results/similarity_comparison_bar.png` — shows the 5 metrics side-by-side
3. `results/ncf_training_loss.png` — proof of GPU training
4. `results/rating_loop.png` — the feedback loop visualization

For Post 2 (visual): Attach 3 images:
1. `results/bias_analysis.png` — coverage/novelty/diversity bars
2. `results/similarity_heatmaps.png` — the 5 heatmaps grid
3. `results/rating_loop.png`

For Post 3 (short): Attach 1 image:
1. `results/ablation_study.png`

---

## Tips for posting

- **Tag DecodeLabs** in the post (and your trainer/mentor if applicable).
- **Tag the tools**: #PyTorch #Python #scikit-learn — these communities engage with technical posts.
- **First 2 lines matter most** — LinkedIn truncates after ~210 chars. Put the hook there.
- **Post on Tuesday–Thursday, 8–10 AM your timezone** — highest engagement window for tech content.
- **Reply to every comment within 2 hours** — LinkedIn's algorithm rewards early engagement.
- **Reshare to your GitHub Projects page** — link the LinkedIn post in the repo's README under "Featured".
