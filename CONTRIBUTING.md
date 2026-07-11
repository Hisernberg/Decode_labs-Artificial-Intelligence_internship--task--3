# Contributing

Contributions are welcome! This is a learning project, so all skill levels are encouraged to participate.

## 🐛 Reporting Issues

If you find a bug or have a feature request:
1. Check [existing issues](../../issues) to avoid duplicates.
2. Open a new issue with:
   - Clear title and description
   - Minimal reproducible example (code + data)
   - Expected vs actual behavior
   - Environment info (Python version, OS, GPU/CPU)

## 🔧 Submitting Pull Requests

1. Fork the repo and create a feature branch:
   ```bash
   git checkout -b feature/my-new-feature
   ```
2. Make your changes. Keep commits focused — one logical change per commit.
3. Test locally:
   ```bash
   pip install -r requirements.txt
   python -c "from src import generate_synthetic_ecommerce; print('ok')"
   ```
4. Run the notebook end-to-end to confirm nothing breaks.
5. Push and open a PR with a clear description of what changed and why.

## 🎯 Areas That Need Help

- [ ] Sequential recommender (SASRec / BERT4Rec) on top of the existing models
- [ ] Multi-objective Pareto-front exploration (accuracy vs diversity)
- [ ] Fairness audit: do recs systematically disadvantage certain categories?
- [ ] Explainability: attach reason codes ("Because you liked X") to each rec
- [ ] Online learning: replace batch refit with streaming updates
- [ ] Streamlit / Gradio front-end for the interactive demo

## 📝 Coding Standards

- **Python style:** PEP 8, type hints where helpful, docstrings on public functions
- **Notebook style:** Each cell should run independently (idempotent); markdown before code; clear section headers
- **Commit messages:** imperative mood ("Add NMF model" not "Added NMF model")
- **No large data files in git** — the notebook regenerates synthetic data; real datasets are downloaded at runtime

## 📜 Code of Conduct

Be kind. Be patient with beginners. Assume good intent. Disagree about code, not people.

---

Thanks for contributing! 🙏
