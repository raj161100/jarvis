import subprocess
import datetime
import json
import urllib.request
import urllib.parse
import psutil

from config import NEWS_API_KEY

try:
    from duckduckgo_search import DDGS
    DDG_AVAILABLE = True
except ImportError:
    DDG_AVAILABLE = False


class ToolHandler:
    """
    Two tiers:
      Offline — time, date, system stats. Always available, instant.
      Online  — weather, news, search, Wikipedia. Degrade gracefully if offline.

    Tool names must match exactly what ParseAndSecureAgent returns in 'tool_name'.
    """

    TOOL_MAP = {
        "time_query":    "_time_query",
        "date_query":    "_date_query",
        "system_status": "_system_status",
        "weather":       "_weather",
        "news":          "_news",
        "web_search":    "_web_search",
        "wikipedia":     "_wikipedia",
    }

    def handle(self, tool_name, entities=None):
        method_name = self.TOOL_MAP.get(tool_name)
        if method_name:
            return getattr(self, method_name)(entities or [])
        return None

    # ── OFFLINE ───────────────────────────────────────────────────────────────

    def _time_query(self, _):
        return f"The time is {datetime.datetime.now().strftime('%I:%M %p')}."

    def _date_query(self, _):
        return f"Today is {datetime.datetime.now().strftime('%A, %d %B %Y')}."

    def _system_status(self, _):
        cpu  = psutil.cpu_percent(interval=0.5)
        ram  = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        temp = self._pi_temp()
        throttle = self._pi_throttle()
        return (
            f"CPU at {cpu} percent. "
            f"RAM at {ram.percent} percent, "
            f"{ram.used // 1024**2} of {ram.total // 1024**2} megabytes used. "
            f"Disk at {disk.percent} percent, {disk.free // 1024**3} gigabytes free. "
            f"Core temperature {temp}, thermal status {throttle}."
        )

    def _pi_temp(self):
        try:
            r = subprocess.run(["vcgencmd", "measure_temp"], capture_output=True, text=True)
            return r.stdout.strip().replace("temp=", "")
        except Exception:
            try:
                with open("/sys/class/thermal/thermal_zone0/temp") as f:
                    return f"{int(f.read()) / 1000:.1f}°C"
            except Exception:
                return "unavailable"

    def _pi_throttle(self):
        try:
            r = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True)
            raw = r.stdout.strip().replace("throttled=", "")
            return "healthy" if raw == "0x0" else f"throttled ({raw})"
        except Exception:
            return "unknown"

    # ── ONLINE — no API key needed ────────────────────────────────────────────

    def _weather(self, entities):
        location = entities[0] if entities else "London"
        try:
            url = f"https://wttr.in/{urllib.parse.quote(location)}?format=j1"
            with urllib.request.urlopen(url, timeout=5) as r:
                data = json.loads(r.read())
            c    = data["current_condition"][0]
            desc = c["weatherDesc"][0]["value"]
            return (
                f"Current weather in {location}: {desc}. "
                f"Temperature {c['temp_C']} degrees Celsius, feels like {c['FeelsLikeC']}. "
                f"Humidity {c['humidity']} percent."
            )
        except Exception:
            return "I couldn't reach the weather service. Check your internet connection."

    def _news(self, entities):
        try:
            if NEWS_API_KEY:
                query = urllib.parse.quote(entities[0] if entities else "technology")
                url = f"https://newsapi.org/v2/top-headlines?q={query}&apiKey={NEWS_API_KEY}&pageSize=3"
                with urllib.request.urlopen(url, timeout=5) as r:
                    articles = json.loads(r.read()).get("articles", [])[:3]
                if not articles:
                    return "No news articles found on that topic."
                return "Top headlines: " + ". ".join(a["title"] for a in articles) + "."
            else:
                # Keyless BBC RSS fallback
                import re
                with urllib.request.urlopen("https://feeds.bbci.co.uk/news/rss.xml", timeout=5) as r:
                    raw = r.read().decode()
                titles = re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", raw)[:4]
                if titles:
                    return "Latest BBC headlines: " + ". ".join(titles[1:]) + "."
                return "Couldn't parse the news feed."
        except Exception:
            return "I couldn't retrieve the news right now."

    def _web_search(self, entities):
        if not DDG_AVAILABLE:
            return "Web search requires: pip install duckduckgo-search"
        query = " ".join(entities) if entities else ""
        if not query:
            return "I need a search query to look something up."
        try:
            with DDGS() as ddg:
                results = list(ddg.text(query, max_results=3))
            if not results:
                return "No results found for that search."
            summary = ". ".join(r["body"][:120] for r in results if r.get("body"))
            return f"Here's what I found: {summary}"
        except Exception as e:
            return f"Search failed: {e}"

    def _wikipedia(self, entities):
        topic = urllib.parse.quote(entities[0] if entities else "")
        if not topic:
            return "What topic would you like me to look up?"
        try:
            req = urllib.request.Request(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}",
                headers={"User-Agent": "JarvisPI/1.0"}
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                data = json.loads(r.read())
            extract   = data.get("extract", "")
            sentences = extract.split(". ")[:2]
            return ". ".join(sentences) + "."
        except Exception:
            return "I couldn't find a Wikipedia article on that topic."
