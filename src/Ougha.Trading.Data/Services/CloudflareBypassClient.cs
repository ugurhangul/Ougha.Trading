using System.Net;
using System.Text.RegularExpressions;
using Jint;

namespace Ougha.Trading.Data.Services;

/// <summary>
/// HTTP client that can bypass Cloudflare protection by solving JS challenges.
/// Similar to Python's cloudscraper library.
/// </summary>
public class CloudflareBypassClient : IDisposable
{
    private readonly HttpClient _httpClient;
    private readonly CookieContainer _cookies;
    private bool _challenged;
    
    private static readonly string[] BrowserUserAgents =
    [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15"
    ];
    
    public CloudflareBypassClient()
    {
        _cookies = new CookieContainer();
        var handler = new HttpClientHandler
        {
            CookieContainer = _cookies,
            AutomaticDecompression = DecompressionMethods.GZip | DecompressionMethods.Deflate | DecompressionMethods.Brotli,
            AllowAutoRedirect = true
        };
        
        _httpClient = new HttpClient(handler);
        
        // Set browser-like headers
        var userAgent = BrowserUserAgents[Random.Shared.Next(BrowserUserAgents.Length)];
        _httpClient.DefaultRequestHeaders.Add("User-Agent", userAgent);
        _httpClient.DefaultRequestHeaders.Add("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8");
        _httpClient.DefaultRequestHeaders.Add("Accept-Language", "en-US,en;q=0.9");
        _httpClient.DefaultRequestHeaders.Add("Accept-Encoding", "gzip, deflate, br");
        _httpClient.DefaultRequestHeaders.Add("Cache-Control", "no-cache");
        _httpClient.DefaultRequestHeaders.Add("Pragma", "no-cache");
        _httpClient.DefaultRequestHeaders.Add("Sec-Ch-Ua", "\"Not_A Brand\";v=\"8\", \"Chromium\";v=\"120\", \"Google Chrome\";v=\"120\"");
        _httpClient.DefaultRequestHeaders.Add("Sec-Ch-Ua-Mobile", "?0");
        _httpClient.DefaultRequestHeaders.Add("Sec-Ch-Ua-Platform", "\"Windows\"");
        _httpClient.DefaultRequestHeaders.Add("Sec-Fetch-Dest", "document");
        _httpClient.DefaultRequestHeaders.Add("Sec-Fetch-Mode", "navigate");
        _httpClient.DefaultRequestHeaders.Add("Sec-Fetch-Site", "none");
        _httpClient.DefaultRequestHeaders.Add("Sec-Fetch-User", "?1");
        _httpClient.DefaultRequestHeaders.Add("Upgrade-Insecure-Requests", "1");
        
        _httpClient.Timeout = TimeSpan.FromSeconds(30);
    }
    
    public async Task<string> GetStringAsync(string url)
    {
        var uri = new Uri(url);
        var response = await _httpClient.GetAsync(url);
        
        // Check for Cloudflare challenge
        if (response.StatusCode == HttpStatusCode.Forbidden || 
            response.StatusCode == HttpStatusCode.ServiceUnavailable)
        {
            var content = await response.Content.ReadAsStringAsync();
            
            if (IsCloudflareChallenge(content))
            {
                Console.WriteLine("[CloudflareBypassClient] Cloudflare challenge detected, solving...");
                await SolveChallenge(uri, content);
                
                // Retry the request
                response = await _httpClient.GetAsync(url);
            }
        }
        
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadAsStringAsync();
    }
    
    private static bool IsCloudflareChallenge(string html)
    {
        return html.Contains("cf-browser-verification") || 
               html.Contains("jschl-answer") ||
               html.Contains("__cf_chl") ||
               html.Contains("_cf_chl_opt");
    }
    
    private async Task SolveChallenge(Uri uri, string html)
    {
        // Wait like a browser would
        await Task.Delay(4000);
        
        // Try to extract and solve the JS challenge
        try
        {
            // Look for the challenge script
            var scriptMatch = Regex.Match(html, @"setTimeout\(function\(\)\s*\{([^}]+)},\s*(\d+)\)", RegexOptions.Singleline);
            
            if (scriptMatch.Success)
            {
                var script = scriptMatch.Groups[1].Value;
                var answer = SolveJsChallenge(script, uri.Host);
                
                if (!string.IsNullOrEmpty(answer))
                {
                    // Extract form action
                    var formMatch = Regex.Match(html, @"action=""([^""]+)""");
                    var rayMatch = Regex.Match(html, @"name=""r""\s+value=""([^""]*)""");
                    var jschlVcMatch = Regex.Match(html, @"name=""jschl_vc""\s+value=""([^""]*)""");
                    var passMatch = Regex.Match(html, @"name=""pass""\s+value=""([^""]*)""");
                    
                    if (formMatch.Success)
                    {
                        var formUrl = new Uri(uri, formMatch.Groups[1].Value);
                        var formData = new FormUrlEncodedContent(new Dictionary<string, string>
                        {
                            ["r"] = rayMatch.Success ? rayMatch.Groups[1].Value : "",
                            ["jschl_vc"] = jschlVcMatch.Success ? jschlVcMatch.Groups[1].Value : "",
                            ["pass"] = passMatch.Success ? passMatch.Groups[1].Value : "",
                            ["jschl_answer"] = answer
                        });
                        
                        await _httpClient.PostAsync(formUrl, formData);
                    }
                }
            }
            
            // Modern Cloudflare might use Turnstile - we can't solve that easily
            // Fall back to just waiting and retrying
            _challenged = true;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[CloudflareBypassClient] Challenge solving failed: {ex.Message}");
        }
    }
    
    private static string SolveJsChallenge(string script, string host)
    {
        try
        {
            var engine = new Engine();
            
            // Add document host length (used in calculations)
            engine.SetValue("hostLength", host.Length);
            
            // Clean up the script for evaluation
            var cleanScript = script
                .Replace("a.value", "result")
                .Replace("t.length", "hostLength");
            
            // Execute and get result
            engine.Execute($"var result = 0; {cleanScript}");
            var result = engine.GetValue("result").AsNumber();
            
            return result.ToString("F10");
        }
        catch
        {
            return "";
        }
    }
    
    public void Dispose()
    {
        _httpClient.Dispose();
    }
}
