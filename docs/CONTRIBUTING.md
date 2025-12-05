# 🤝 Contributing to Automatos AI

> **"We're building the operating system for AI agents. We need your help."**

First off, thank you for considering contributing to Automatos AI! It's people like you that make open source such a powerful force.

---

## 🌟 Ways to Contribute

You don't need to be an AI expert to help out. Here are high-impact ways to contribute:

### 1. 🛠️ Build Tools (Best for Beginners)
The easiest way to add value is to add a new tool to the registry.
- **Idea:** Add a tool for Jira, Trello, GitHub, or your favorite API.
- **Guide:** See [Developer Guide > Adding a Tool](DEVELOPER_GUIDE.md#tutorial-1-adding-a-new-tool).

### 2. 🐛 Fix Bugs
Find a bug? Fix it!
- Look for issues labeled `good first issue`.
- Check `FIXME` or `TODO` comments in the code.

### 3. 📝 Improve Documentation
Docs are never perfect.
- Fix typos.
- Add examples.
- Write tutorials.
- Translate docs.

### 4. 🧠 Enhance Core Modules
Ready for a challenge?
- Improve the RAG pipeline.
- Optimize the memory system.
- Add new agent coordination patterns.

---

## 🚀 Pull Request Process

### 1. Fork & Clone
Fork the repo to your account and clone it locally.

### 2. Create a Branch
Use a descriptive name:
```bash
git checkout -b feat/add-jira-tool
# or
git checkout -b fix/memory-leak-in-stream
```

### 3. Make Changes
- Follow the [Developer Guide](DEVELOPER_GUIDE.md).
- Keep changes focused (one feature per PR).
- Add tests for new code.

### 4. Test & Lint
Before pushing, ensure everything is green:
```bash
# Run tests
pytest

# Check style
ruff check .
```

### 5. Push & Open PR
Push to your fork and open a Pull Request against `main`.
- **Title**: Clear and descriptive (e.g., "feat: Add Jira integration tool").
- **Description**: Explain *what* you did and *why*.
- **Screenshots**: If you changed the UI, show us!

---

## 🎨 Style Guidelines

- **Python**: We follow PEP 8. Use `ruff` to format.
- **Types**: All function arguments and return values must have type hints.
- **Docstrings**: Every public function/class needs a docstring.
- **Async**: Use `async/await` for all I/O operations.

---

## 📜 Code of Conduct

**Be kind. Be respectful. Be constructive.**

We are committed to providing a friendly, safe, and welcoming environment for all, regardless of gender, sexual orientation, disability, ethnicity, religion, or similar personal characteristic.

Harassment of any kind will not be tolerated.

---

## 🏆 Recognition

We love our contributors!
- **All contributors** are listed in the repository.
- **Major contributors** get shout-outs in our release notes.
- **Consistent contributors** may be invited to join the core team.

---

**Questions?**
Join our [Discord](https://discord.gg/automatos) and ask in the `#dev-chat` channel.

**Happy Hacking!** 💻
