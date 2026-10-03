Output docs/raw.mp4
Set Shell "bash"
Set FontSize 14
Set Width 1180
Set Height 900
Set Padding 16
Set Framerate 24
Set Theme "Catppuccin Mocha"
Set TypingSpeed 45ms
Hide
Type "cd docs/.sp/afdemo && uv venv -q --python 3.12 .venv && source .venv/bin/activate && export UV_FIND_LINKS=$PWD/../wheels && alias pip='uv pip' && export PS1='$ ' && clear" Enter
Sleep 12s
Show
Sleep 600ms
Type "pip install alphafinch" Sleep 300ms Enter
Wait+Screen@180s /alphafinch==0.1.0/
Sleep 1.2s
Type "alphafinch evolve --market us --generations 3" Sleep 300ms Enter
Sleep 150s
Sleep 3s
