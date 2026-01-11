# AI Skills Hub - Documentation Site

This directory contains the MkDocs-based documentation site for AI Skills Hub, a free AI learning platform.

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

### Local Development

1. **Install dependencies:**
   ```bash
   cd aitutor
   pip install -r requirements.txt
   ```

2. **Start the development server:**
   ```bash
   mkdocs serve
   ```

3. **View the site:**
   Open your browser to `http://localhost:8000`

   The site will automatically reload when you save changes to any files.

### Building the Site

To build the static site files:

```bash
mkdocs build
```

The built site will be in the `site/` directory.

## Project Structure

```
aitutor/
├── docs/                      # Documentation content
│   ├── index.md              # Landing page
│   ├── courses/              # Course tracks
│   │   ├── foundation/       # Foundation track
│   │   ├── core/             # Core track
│   │   └── advanced/         # Advanced track
│   ├── projects/             # Portfolio projects
│   ├── resources/            # Learning resources
│   ├── stylesheets/          # Custom CSS
│   │   └── custom.css
│   └── javascripts/          # Custom JS (optional)
├── notebooks/                # Jupyter/Colab notebooks
│   └── quickstart.ipynb      # Quick start tutorial
├── mkdocs.yml               # MkDocs configuration
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Content Guidelines

### Adding New Pages

1. Create a new Markdown file in the appropriate directory under `docs/`
2. Update the `nav` section in `mkdocs.yml` to include the new page
3. Follow the existing content structure and style

### Writing Course Content

Each course module should include:
- Clear learning objectives
- Interactive code examples
- Links to Colab/Kaggle notebooks
- Hands-on exercises
- Assessment quizzes

### Code Blocks

Use fenced code blocks with language specification:

````markdown
```python
import torch
print("Hello, AI!")
```
````

### Linking to Notebooks

Use Colab badges:
```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rajgupt/ai-for-builders/blob/main/aitutor/notebooks/your-notebook.ipynb)
```

## Deployment

### Automatic Deployment

The site automatically deploys to GitHub Pages when you push to the `main` branch.

The deployment workflow is configured in `.github/workflows/deploy.yml`.

### Manual Deployment

To manually deploy to GitHub Pages:

```bash
mkdocs gh-deploy
```

This will:
1. Build the site
2. Push the built files to the `gh-pages` branch
3. GitHub Pages will serve the site

## Site URL

After deployment, the site will be available at:
- **Production:** `https://rajgupt.github.io/ai-for-builders/`

## Customization

### Theme Configuration

Edit `mkdocs.yml` to customize:
- Color scheme
- Navigation structure
- Enabled features
- Social links
- Analytics

### Custom Styling

Add custom CSS in `docs/stylesheets/custom.css`.

The file is already configured with:
- Hero section styling
- Course card layouts
- CTA button styles
- Responsive design

### Custom JavaScript

Add custom JavaScript in `docs/javascripts/` and reference in `mkdocs.yml`.

## MkDocs Material Features

This site uses Material for MkDocs with the following features:

- **Dark theme** optimized for developers
- **Instant navigation** for SPA-like experience
- **Code copy buttons** for easy code sharing
- **Search** with autocomplete
- **Responsive design** for mobile and desktop
- **SEO optimization** with meta tags

## Development Workflow

1. **Create a new branch** for your changes
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** to the documentation

3. **Preview locally** with `mkdocs serve`

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "Description of changes"
   ```

5. **Push to GitHub**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request** on GitHub

7. **After merge to main**, the site will automatically deploy

## Troubleshooting

### Site not loading locally

- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check that you're in the `aitutor` directory
- Try clearing your browser cache

### Build errors

- Validate your YAML syntax in `mkdocs.yml`
- Check for broken internal links
- Ensure all referenced files exist

### Deployment failures

- Check GitHub Actions logs in the repository
- Verify GitHub Pages is enabled in repository settings
- Ensure the `gh-pages` branch exists

## Resources

- **MkDocs Documentation:** https://www.mkdocs.org/
- **Material for MkDocs:** https://squidfunk.github.io/mkdocs-material/
- **Markdown Guide:** https://www.markdownguide.org/

## Contributing

We welcome contributions! Please:

1. Read the project instructions in `/CLAUDE.md`
2. Follow the content guidelines above
3. Test your changes locally before submitting
4. Create clear, descriptive commit messages

## License

This project is licensed under the MIT License. All course content is free and open source.

---

**Questions?** Open an issue on [GitHub](https://github.com/rajgupt/ai-for-builders/issues) or start a [discussion](https://github.com/rajgupt/ai-for-builders/discussions).
