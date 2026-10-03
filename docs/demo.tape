Output docs/demo.gif
Set Shell "bash"
Set FontSize 13
Set Width 1500
Set Height 640
Set Padding 18
Set Theme "Catppuccin Mocha"
Set PlaybackSpeed 2
Set TypingSpeed 40ms
Hide
Type "source .venv/bin/activate && clear" Enter
Show
Type "alphafinch evolve --market us --provider claude-code --generations 3 --islands 4 --offspring 3 --out runs/gif" Sleep 500ms Enter
Wait+Screen@600s /Report:/
Sleep 4s
