# Task 01: Developer-Focused Landing Page Setup

## Overview

Create a minimalistic, high-performance landing page for **AI Skills Hub** targeting hands-on developers and technical practitioners. The page should communicate value instantly and prioritize navigation to actionable content.

---

## Target Audience

**Primary:** Hands-on developers, data scientists, ML engineers who want to:
- Build AI projects immediately
- Access free GPU compute (Colab/Kaggle)
- Create portfolio-worthy projects
- Learn by coding, not by watching

**NOT targeting:** General AI enthusiasts, non-technical learners, passive consumers

---

## Design Principles

### 1. **Code-First Messaging**
- Lead with runnable code examples
- Show "Open in Colab" badges prominently
- Display actual code snippets, not marketing fluff

### 2. **Minimal Chrome, Maximum Content**
- No hero images or stock photos
- Terminal/IDE aesthetic preferred
- Monospace fonts for technical credibility

### 3. **Speed & Navigation**
- Sub-2-second load time
- Sticky navigation bar
- Direct links to course modules (no unnecessary clicks)

### 4. **Technical Credibility**
- Show GitHub stars/forks count
- Display tech stack icons (Python, PyTorch, TensorFlow)
- Link to public notebooks immediately

---

## Landing Page Structure

### Header Section
```
┌─────────────────────────────────────────────────┐
│ [Logo] AI Skills Hub    Courses | Projects | Docs│
│                                                   │
│ Free AI Education for Developers                 │
│ Learn ML/DL with free GPUs • Zero setup required │
│                                                   │
│ [Browse Courses →] [Open Colab Notebook →]       │
│                                                   │
│ ✓ 100% Free  ✓ GPU Included  ✓ Portfolio Projects│
└─────────────────────────────────────────────────┘
```

**Key Elements:**
- One-line value proposition (no paragraph)
- Two primary CTAs: Browse Courses + Try Sample Notebook
- Three trust badges (succinct, icon-based)

---

### Course Track Overview (3-Column Grid)

```
┌───────────────┬───────────────┬───────────────┐
│ Foundation    │ Core          │ Advanced      │
├───────────────┼───────────────┼───────────────┤
│ Python for AI │ Deep Learning │ Transformers  │
│ ML Basics     │ Computer Vision│ LLMs & GenAI  │
│ 7 weeks       │ 10 weeks      │ 14 weeks      │
│ [Start →]     │ [Start →]     │ [Start →]     │
└───────────────┴───────────────┴───────────────┘
```

**Requirements:**
- Clean grid layout (Material theme cards)
- Week count visible (sets expectations)
- Direct navigation to first lesson
- No course descriptions (keep it scannable)

---

### Quick Start Code Sample

Show a 10-line code snippet that:
1. Runs in Colab/Kaggle
2. Produces visible output (plot, prediction)
3. Uses popular libraries (PyTorch/TensorFlow)
4. Has "Copy" button and "Open in Colab" badge

**Example:**
```python
# Train your first neural network in 10 lines
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
# Full tutorial: [link]
```

**Placement:** Below course tracks, before footer

---

### Footer (Minimal)

```
[GitHub] [Discussions] [Contribute]
Built with MkDocs • Free Forever • MIT License
```

---

## Technical Implementation

### File Structure
```
docs/
├── index.md                 # Landing page content
├── stylesheets/
│   └── custom.css           # Developer-focused styling
└── javascripts/
    └── landing.js           # CTA tracking (optional)
```

### mkdocs.yml Configuration

```yaml
site_name: AI Skills Hub
site_description: Free AI education for developers with GPU access
site_url: https://rajgupt.github.io/ai-for-builders

theme:
  name: material
  palette:
    scheme: slate                # Dark theme for developers
    primary: deep purple
    accent: cyan
  font:
    text: Inter
    code: JetBrains Mono         # Monospace for code blocks
  features:
    - navigation.instant         # SPA-like navigation
    - navigation.tracking        # URL updates on scroll
    - navigation.tabs            # Top-level tabs
    - search.suggest             # Search autocomplete
    - content.code.copy          # Copy button on code blocks
    - content.code.annotate      # Code annotations

extra_css:
  - stylesheets/custom.css

extra:
  social:
    - icon: fontawesome/brands/github
      link: https://github.com/rajgupt/ai-for-builders
  analytics:
    provider: google
    property: G-XXXXXXXXXX       # Add GA4 tracking ID

nav:
  - Home: index.md
  - Courses:
    - Foundation: courses/foundation/index.md
    - Core: courses/core/index.md
    - Advanced: courses/advanced/index.md
  - Projects: projects/index.md
  - Resources: resources/index.md
```

---

## Custom Styling (custom.css)

```css
/* Developer-focused overrides */
:root {
  --md-primary-fg-color: #7c4dff;
  --md-accent-fg-color: #00e5ff;
}

/* Hero section */
.hero-section {
  text-align: center;
  padding: 4rem 2rem;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.hero-section h1 {
  font-size: 2.5rem;
  margin-bottom: 1rem;
  font-weight: 700;
}

.hero-section p {
  font-size: 1.25rem;
  margin-bottom: 2rem;
  opacity: 0.9;
}

/* CTA buttons */
.cta-button {
  display: inline-block;
  padding: 0.75rem 1.5rem;
  margin: 0.5rem;
  background: white;
  color: #667eea;
  border-radius: 4px;
  text-decoration: none;
  font-weight: 600;
  transition: transform 0.2s;
}

.cta-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}

/* Course cards */
.course-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 2rem;
  margin: 3rem 0;
}

.course-card {
  border: 1px solid var(--md-default-fg-color--lightest);
  border-radius: 8px;
  padding: 1.5rem;
  transition: border-color 0.2s;
}

.course-card:hover {
  border-color: var(--md-accent-fg-color);
}

/* Code snippet section */
.code-sample {
  background: #1e1e1e;
  padding: 2rem;
  border-radius: 8px;
  margin: 3rem 0;
}

/* Trust badges */
.trust-badges {
  display: flex;
  justify-content: center;
  gap: 2rem;
  margin-top: 2rem;
  font-size: 0.9rem;
  opacity: 0.8;
}

.trust-badges span {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

/* Minimize footer */
.md-footer {
  padding: 1rem 0;
}
```

---

## Content for index.md

```markdown
---
title: AI Skills Hub - Free AI Education for Developers
description: Learn ML/DL with free GPUs. Zero setup. Portfolio-worthy projects.
---

<div class="hero-section">
  <h1>AI Skills Hub</h1>
  <p>Free AI education for developers • GPU included • Zero setup</p>

  <a href="courses/foundation/" class="cta-button">Browse Courses →</a>
  <a href="https://colab.research.google.com/github/rajgupt/ai-for-builders/blob/main/notebooks/sample.ipynb" class="cta-button">Try Sample Notebook →</a>

  <div class="trust-badges">
    <span>✓ 100% Free</span>
    <span>✓ Free GPU Access</span>
    <span>✓ Portfolio Projects</span>
  </div>
</div>

## Learning Paths

<div class="course-grid">
  <div class="course-card">
    <h3>Foundation Track</h3>
    <ul>
      <li>Python for AI</li>
      <li>Math Essentials</li>
      <li>Introduction to ML</li>
    </ul>
    <p><strong>Duration:</strong> 7 weeks</p>
    <a href="courses/foundation/">Start Learning →</a>
  </div>

  <div class="course-card">
    <h3>Core Track</h3>
    <ul>
      <li>Deep Learning Fundamentals</li>
      <li>Computer Vision</li>
      <li>NLP Basics</li>
    </ul>
    <p><strong>Duration:</strong> 10 weeks</p>
    <a href="courses/core/">Start Learning →</a>
  </div>

  <div class="course-card">
    <h3>Advanced Track</h3>
    <ul>
      <li>Transformers & LLMs</li>
      <li>Generative AI</li>
      <li>MLOps & Production</li>
    </ul>
    <p><strong>Duration:</strong> 14 weeks</p>
    <a href="courses/advanced/">Start Learning →</a>
  </div>
</div>

## Quick Start: Your First Neural Network

Train a neural network in 10 lines of code. Click "Open in Colab" to run with free GPU:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rajgupt/ai-for-builders/blob/main/notebooks/quickstart.ipynb)

```python
import torch
import torch.nn as nn

# Define a simple neural network
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Ready to train! See full tutorial →
```

[View Full Quickstart Tutorial →](courses/foundation/quickstart/)

## Why AI Skills Hub?

| Feature | AI Skills Hub | Traditional Courses |
|---------|---------------|---------------------|
| Cost | Free Forever | $50-500 per course |
| Setup | Zero (browser-based) | Install dependencies |
| GPU Access | Included (Colab/Kaggle) | Buy cloud credits |
| Projects | Portfolio-ready | Assignments only |
| Support | Community-driven | Limited office hours |

## Built for Developers

- **No Fluff:** Code-first approach, theory second
- **Open Source:** All content on [GitHub](https://github.com/rajgupt/ai-for-builders)
- **Proven Tools:** PyTorch, TensorFlow, Hugging Face
- **Real Projects:** Deployable models for your portfolio
- **Global Access:** Works anywhere, no geo-restrictions

---

**Ready to start?** [Browse all courses →](courses/) or [join our community →](https://github.com/rajgupt/ai-for-builders/discussions)
```

---

## Performance Checklist

- [ ] Lighthouse score > 90 (Performance)
- [ ] First Contentful Paint < 1.5s
- [ ] No external dependencies (self-host fonts if needed)
- [ ] Lazy-load images below the fold
- [ ] Minified CSS/JS (via mkdocs-minify-plugin)
- [ ] Responsive design tested (mobile, tablet, desktop)

---

## A/B Testing Considerations (Future)

Track conversion rates for:
1. **CTA clicks:** "Browse Courses" vs "Try Sample Notebook"
2. **Track selection:** Which learning path gets most clicks?
3. **Code sample engagement:** Do users click "Open in Colab"?

Use Google Analytics 4 events to measure.

---

## Success Metrics (Week 1-2)

- **Traffic:** 500 unique visitors
- **Engagement:** Avg. session duration > 3 minutes
- **Conversion:** 20% click-through to courses
- **Bounce rate:** < 65%
- **Colab opens:** At least 50 notebook launches

---

## Dependencies

```
# requirements.txt for local MkDocs development
mkdocs==1.5.3
mkdocs-material==9.5.3
mkdocs-minify-plugin==0.7.1
pymdown-extensions==10.7
```

Install:
```bash
pip install -r requirements.txt
mkdocs serve  # Preview at http://localhost:8000
```

---

## Deployment

Automated via GitHub Actions (`.github/workflows/deploy.yml`):

```yaml
name: Deploy MkDocs to GitHub Pages

on:
  push:
    branches:
      - main

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: 3.x
      - run: pip install -r requirements.txt
      - run: mkdocs gh-deploy --force
```

---

## Next Steps After Landing Page

1. Create placeholder pages for all 3 course tracks
2. Write first lesson: "Python for AI - Setup & Basics"
3. Create sample Colab notebook for "Quick Start" section
4. Set up Google Analytics 4
5. Submit to Google Search Console
6. Share on Reddit (r/learnmachinelearning)

---

## Notes

- **Design Philosophy:** "If a developer can't understand it in 10 seconds, simplify it."
- **Content Over Design:** Prioritize fast loading and clear navigation over fancy animations
- **Mobile-First:** Many learners in developing countries use mobile devices
- **Accessibility:** Ensure high contrast, keyboard navigation, screen reader support

---

**Task Owner:** Rajat Gupta
**Estimated Effort:** 4-6 hours (implementation) + 2 hours (testing)
**Priority:** P0 (blocks all other content creation)
**Status:** Not Started
**Dependencies:** MkDocs setup, GitHub Pages configuration
