# AI Skills Hub - Project Instructions

## Project Overview

**AI Skills Hub** is a free, open-source AI learning platform designed to make AI education accessible to everyone worldwide. The platform uses static site generators (MkDocs + Material Theme) hosted on GitHub Pages, with all courses designed around free compute resources (Google Colab, Kaggle).

### Core Mission
- Provide 100% free AI education with no paywalls
- Eliminate barriers: zero setup, runs in browser, free GPUs
- Build practical, portfolio-worthy projects
- Foster a global learning community

## Technology Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| Static Site Generator | MkDocs + Material Theme | Documentation-focused, beautiful UI |
| Hosting | GitHub Pages | Free SSL, custom domain support |
| Code Execution | Google Colab + Kaggle | Free GPU access for learners |
| Version Control | GitHub | Collaborative editing |
| Analytics | Google Analytics 4 | Traffic insights |

## Repository Structure

```
ai-for-builders/
├── aitutor/
│   ├── docs/               # MkDocs content
│   │   ├── index.md
│   │   ├── courses/        # Course modules
│   │   ├── projects/       # Hands-on projects
│   │   ├── blog/           # SEO-optimized content
│   │   └── resources/      # Additional materials
│   ├── notebooks/          # Jupyter/Colab notebooks
│   ├── plan/               # Strategic & execution plans
│   └── mkdocs.yml          # Site configuration
├── .github/
│   └── workflows/          # CI/CD automation
└── CLAUDE.md              # This file
```

## Content Strategy

### Course Architecture

**Foundation Track (Beginner)**
- Python for AI (2 weeks)
- Math Essentials (2 weeks)
- Introduction to ML (3 weeks)
- Hands-on Projects (2 weeks)

**Core Track (Intermediate)**
- Deep Learning Fundamentals (4 weeks)
- Computer Vision (3 weeks)
- NLP Basics (3 weeks)
- Model Deployment (2 weeks)

**Advanced Track**
- Transformers & LLMs (4 weeks)
- Generative AI (3 weeks)
- MLOps & Production (3 weeks)
- Capstone Project (4 weeks)

### Course Module Template

Each module must include:
1. **Concept Page** - Theory with visuals
2. **Interactive Colab** - Hands-on coding with GPU support
3. **Quiz/Assessment** - Knowledge verification
4. **Mini-Project** - Practical application
5. **Resources** - Further reading links
6. **Discussion** - GitHub Discussions integration

## Development Guidelines

### When Creating Course Content

1. **Write in Markdown** - All content in `.md` files under `docs/`
2. **Link to Colab/Kaggle** - Every coding lesson needs runnable notebooks
3. **Test on Free GPUs** - Verify all code runs on Colab/Kaggle free tier
4. **SEO-Optimize** - Use descriptive headers, keywords, alt text
5. **Mobile-First** - Content must be readable on smartphones
6. **Include Badges** - Add "Open in Colab" and "Open in Kaggle" buttons

### Notebook Guidelines

```python
# Template header for all notebooks:
"""
Course: [Course Name]
Lesson: [Lesson Name]
Platform: AI Skills Hub
License: MIT
GPU Required: [Yes/No]
Estimated Runtime: [X minutes]
"""
```

### Writing Style

- **Beginner-friendly**: Assume no prior knowledge
- **Practical-first**: Code before theory
- **Globally accessible**: Simple English, avoid idioms
- **Action-oriented**: Every lesson builds something
- **Portfolio-ready**: Projects suitable for GitHub/resume

### SEO Best Practices

Target keywords:
- "how to learn machine learning for free"
- "pytorch tutorial for beginners"
- "google colab gpu tutorial"
- "machine learning project ideas"
- "nlp tutorial python"

Structure:
- Use H1, H2, H3 hierarchically
- Include meta descriptions
- Add alt text to all images
- Internal linking between lessons
- External backlinks to notebooks

## Traffic & Growth Strategy

### Content Marketing Channels

**Priority 1: Owned Media**
- Course website blog
- GitHub README files with backlinks
- Branded Colab notebooks

**Priority 2: Earned Media**
- Reddit (r/learnmachinelearning, r/artificial)
- Hacker News
- Dev.to cross-posting
- Medium (with canonical links)
- YouTube tutorials

**Priority 3: Community**
- GitHub Discussions
- Discord server
- Kaggle community

### Growth Targets

| Phase | Timeline | Monthly Visitors | Strategy |
|-------|----------|------------------|----------|
| Launch | Month 1-2 | 500 | Initial content, social sharing |
| Growth | Month 3-6 | 5,000 | SEO ranking, community building |
| Scale | Month 7-12 | 20,000 | Content expansion, backlinks |
| Mature | Month 13-24 | 50,000+ | Brand recognition, word-of-mouth |

## Monetization (Future)

### Phase 1 (Month 1-6): Build Audience
- All content FREE
- Focus on traffic and email list
- Goal: 10,000 monthly visitors

### Phase 2 (Month 7-12): Soft Monetization
- Affiliate marketing (cloud credits, books)
- GitHub Sponsors
- Donations (Ko-fi)

### Phase 3 (Month 13+): Premium Offerings
- Paid certificates
- Premium projects
- 1:1 mentorship
- Corporate training

**Important**: Core education remains free forever.

## Working with Claude Code

### Common Tasks

**Adding a new course module:**
1. Create folder under `docs/courses/[module-name]/`
2. Write lesson.md files using the module template
3. Create corresponding notebooks in `notebooks/[module-name]/`
4. Update `mkdocs.yml` navigation
5. Test locally: `mkdocs serve`
6. Deploy: Push to main (auto-deploys via GitHub Actions)

**Writing blog content:**
1. Create `.md` file in `docs/blog/`
2. Include SEO-optimized title and meta description
3. Use target keywords naturally
4. Link to relevant course modules
5. Add "Read Next" suggestions

**Creating interactive notebooks:**
1. Write in Jupyter/Colab format
2. Add template header with course info
3. Test on Colab free tier (GPU if needed)
4. Test on Kaggle free tier
5. Upload to `notebooks/` directory
6. Link from course page with badges

### File Conventions

- **Course content**: `docs/courses/[track]/[module]/lesson-[number].md`
- **Notebooks**: `notebooks/[module]/lesson-[number].ipynb`
- **Blog posts**: `docs/blog/YYYY-MM-DD-title.md`
- **Project guides**: `docs/projects/[project-name]/README.md`

### Code Standards

- Python code follows PEP 8
- Notebooks have clear markdown explanations
- All code is commented for beginners
- Include expected outputs in notebooks
- Add requirements.txt for local setup

## Key Principles

1. **Always Free**: No paywalls for core education
2. **Beginner-Focused**: Assume zero prior knowledge
3. **Hands-On**: Every lesson has runnable code
4. **Portfolio-Building**: Projects suitable for resumes
5. **Community-Driven**: Open to contributions
6. **Global Access**: Works on any device, any connection
7. **SEO-First**: Content designed to be discovered

## Success Metrics

Track these KPIs:
- Monthly unique visitors
- Email subscribers
- GitHub stars
- Course completions
- Bounce rate (target < 60%)
- Session duration (target > 4 minutes)
- Colab/Kaggle notebook opens

## Account Details

- GitHub: https://github.com/rajgupt
- Repository: ai-for-builders

## Next Steps

1. Review `/aitutor/plan/strategic_plan.md` for full strategy
2. Set up MkDocs with Material theme
3. Create first course module (Python for AI)
4. Launch landing page with email capture
5. Begin content marketing campaigns

---

**Remember**: The goal is to make AI education accessible to anyone, anywhere, with zero barriers to entry.