' Starts Blocky's Linux (WSL distro "blocky") hidden at Windows login and keeps it running.
' Blocky's services start automatically inside it (systemd).
' Install: copy this file into your Startup folder (Win+R -> shell:startup).
' Remove:  delete it from that folder.
Set shell = CreateObject("WScript.Shell")
shell.Run "wsl.exe -d blocky -- sleep infinity", 0, False
