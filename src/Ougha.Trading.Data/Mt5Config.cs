namespace Ougha.Trading.Data;

/// <summary>
/// MT5 configuration loaded from appsettings.
/// </summary>
public class Mt5Config
{
    public string Login { get; set; } = "";
    public string Password { get; set; } = "";
    public string Server { get; set; } = "";
    public string PythonDllPath { get; set; } = @"C:\Users\Ougha\AppData\Local\Programs\Python\Python312\python312.dll";
}
