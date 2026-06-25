' 根据 config 的 gui.show_console 决定有无 CMD 窗口；支持传入配置文件名
' 用法: 双击 run_gui.vbs
'       或 wscript run_gui.vbs config_5557.yaml

Option Explicit

Dim fso, shell, dir, configPath, line, showConsole
Dim pythonExe, pythonwExe, launchScript, configArg, launchCmd

Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
dir = fso.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = dir

pythonExe = dir & "\.venv\Scripts\python.exe"
pythonwExe = dir & "\.venv\Scripts\pythonw.exe"
launchScript = dir & "\launch_gui.py"

configArg = ""
If WScript.Arguments.Count > 0 Then
    configArg = " --config """ & WScript.Arguments(0) & """"
    configPath = dir & "\" & WScript.Arguments(0)
Else
    configPath = dir & "\config.yaml"
End If

showConsole = False
If fso.FileExists(configPath) Then
    Dim tf
    Set tf = fso.OpenTextFile(configPath, 1, False)
    Do Until tf.AtEndOfStream
        line = Trim(tf.ReadLine)
        If InStr(line, "show_console:") > 0 Then
            If InStr(LCase(line), "true") > 0 Then
                showConsole = True
            End If
            Exit Do
        End If
    Loop
    tf.Close
End If

launchCmd = """" & launchScript & """" & configArg

If showConsole Then
    shell.Run """" & pythonExe & """ " & launchCmd, 1, False
Else
    shell.Run """" & pythonwExe & """ " & launchCmd, 0, False
End If
