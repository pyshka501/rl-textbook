# Contributing

Thank you for considering contributing to this project! Here's how you can help.

## Ways to Contribute

- **Fix typos or errors** in the book text or code notebooks
- **Improve explanations** — add intuition, examples, or diagrams
- **Add exercises** with solutions
- **Translate** the book into another language
- **Improve notebooks** — add experiments, visualisations, or better code

## Getting Started

1. Fork the repository
2. Create a feature branch: `git checkout -b fix/typo-ch03`
3. Make your changes
4. Test: compile the book (`cd book && latexmk -pdf main.tex`) and run notebooks
5. Commit with a clear message: `git commit -m "Fix equation 3.5 typo in ch03"`
6. Push and open a Pull Request

## Book (LaTeX)

- Source files are in `book/`
- Follow the style guide in `book/style_guide.md`
- Use `\,---\,` for em-dashes
- Use the custom commands defined in `main.tex` (e.g., `\E`, `\Prob`, `\cS`)

## Notebooks

- One notebook per chapter in `notebooks/`
- Must run in Google Colab / Kaggle (free tier)
- Include `!pip install` cells for dependencies
- Use English for code and comments
- Include visualisations with matplotlib

## Translations

- Add translations under `translations/<lang_code>/`
- Follow the same chapter structure as the English version
- See `translations/README.md` for details

## Code of Conduct

Be respectful. We welcome contributors of all backgrounds and experience levels.

## License of Contributions

By submitting a pull request, patch, translation, or any other contribution, you agree that your contribution is licensed under the same terms as the rest of the project: **CC BY-NC-ND 4.0** (see [LICENSE](LICENSE)). Printing for sale, commercial publishing, and derivative works require prior written permission from the author.
