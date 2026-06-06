# Fixed Prompt Templates for PSRM
This file documents the fixed prompt templates used by the Posterior Semantic Refinement Module (PSRM). These prompts are used only to construct CLIP semantic anchors for latent posterior refinement. They are fixed before evaluation and are not generated or modified by any external language model during inference.
## Prompt Format
The implementation supports a prompt table with the following format:
```text
content_file|style_file|content_prompt|style_prompt|negative_prompt
```
The wildcard symbol `*` indicates that the same prompt entry is shared across all content or style images.
## Template Definitions
### Content Prompt Template
```text
a natural image preserving the main subject and spatial layout
```
This template provides a weak semantic anchor for content preservation. It avoids detailed image captions and does not introduce object-level information beyond the content image itself.
### Style Prompt Template
```text
an artwork preserving the reference color distribution, texture pattern, and brushstroke characteristics
```
This template provides a weak semantic anchor for style consistency. It describes generic style-related attributes and does not introduce additional semantic objects.
### Negative Prompt Template
```text
low quality, distorted structure, semantic drift, object deformation, inconsistent content
```
This template is shared across all samples and is used to suppress undesirable semantic drift and structural degradation.
## Default Prompt Entry
The default prompt entry used for PSRM is:
```text
*|*|a natural image preserving the main subject and spatial layout|an artwork preserving the reference color distribution, texture pattern, and brushstroke characteristics|low quality, distorted structure, semantic drift, object deformation, inconsistent content
```
