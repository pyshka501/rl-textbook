# RL Book Style Guide for Chapter Authors

## Chapter Structure
Each chapter must follow this exact pattern:

1. `\chapter{Title}` + `\label{ch:label}`
2. Opening quote (italic, in `\begin{quote}...\end{quote}`)
3. `\bigskip` + introductory paragraph(s) explaining the chapter's purpose
4. Multiple `\section{...}` with `\label{sec:...}`
5. Closing `\section*{Exercises}` with `\addcontentsline{toc}{section}{Exercises}`

## LaTeX Conventions
- Use custom commands: `\E` (expectation), `\Var`, `\Prob`, `\R`, `\N`, `\cA`, `\cS`, `\cR`, `\cT`, `\cN`, `\ind` (indicator), `\argmax`, `\argmin`, `\KL{p}{q}`
- Theorem environments: `definition`, `example`, `exercise`, `theorem`, `lemma`, `proposition`, `corollary`, `remark`
- All are numbered `[chapter]`
- Use `\index{...}` liberally for key terms
- Algorithms: `\begin{algorithm}[ht]` with `algorithm2e` package (`\KwIn`, `\For`, `\If`, etc.)
- Tables: use `booktabs` (`\toprule`, `\midrule`, `\bottomrule`)
- TikZ diagrams with: `positioning,arrows,arrows.meta,shapes.geometric,calc`
- Lists: `\begin{itemize}[leftmargin=*]` or `\begin{description}[leftmargin=!,labelwidth=3.5cm]`
- Cross-references: `Chapter~\ref{ch:label}`, `Section~\ref{sec:label}`, `Equation~\eqref{eq:label}`

## Content Style (Sutton & Barto textbook style)
- **Detailed prose**: Every concept must be explained in full sentences, not bullet points
- **Worked examples**: At least 2-3 per chapter with concrete numbers (gridworld, simple MDPs, etc.)
- **Intuitive explanations**: Before formal definitions, explain the concept informally
- **Proofs**: Include full proofs for key results
- **Diagrams**: TikZ diagrams for key concepts (backup diagrams, algorithm flow, etc.)
- **Comparison tables**: Compare related algorithms/methods
- **Bias-variance analysis**: Discuss bias, variance, convergence wherever applicable
- **Historical notes**: Brief mentions of who introduced what and when
- **Exercises**: 6-10 exercises ranging from computation to proof

## Key Cross-Reference Labels
- ch:intro (Chapter 1), ch:bandits (Chapter 2), ch:mdp (Chapter 3)
- ch:dp (Chapter 4), ch:mc (Chapter 5), ch:td (Chapter 6)
- ch:value-approx (Chapter 7), ch:pg (Chapter 8), ch:actor-critic (Chapter 9)
- ch:lm-agent (Chapter 10), ch:practical-rlhf (Chapter 14)
- ch:agents (Chapter 15), ch:open (Chapter 17)

## Target Length
Each chapter should be 400-600 lines of LaTeX (substantial textbook chapter).
