# Portable report delivery blocker

The canonical report input is available in `artifact.json`, but the packaged
HTML report could not be generated in this environment because the required
`node` executable is not installed (`/bin/bash: node: command not found`).

No alternate hand-written HTML renderer was used because the Data Analytics
report contract requires the canonical portable artifact builder.
