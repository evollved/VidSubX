#ifndef MyAppVersion
  #define MyAppVersion "0.1"
#endif
#ifndef OutputBaseFilename
  #define OutputBaseFilename "VSX-From-Inno-IDE"
#endif


#define MyAppName "VidSubX"
#define MyAppPublisher "Victor N"
#define MyAppURL "https://github.com/voun7/VidSubX"
#define MyAppExeName "VSX.exe"

[Setup]
AppId={{1BF98D9C-B728-4938-82FC-18EA0B08BB0E}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
VersionInfoVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
UninstallDisplayIcon={app}\{#MyAppExeName}
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
DisableProgramGroupPage=yes
OutputDir=..
OutputBaseFilename="{#OutputBaseFilename}"
SetupIconFile=vsx.ico
SolidCompression=yes
WizardStyle=modern dynamic

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: checkedonce

[Files]
Source: "..\dist\VSX\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\dist\VSX\_internal\*"; DestDir: "{app}\_internal"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{userappdata}\{#MyAppName}"
Type: filesandordirs; Name: "{localappdata}\{#MyAppName}"
