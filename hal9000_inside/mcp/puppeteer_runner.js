const puppeteer = require('puppeteer');

(async () => {
    const action = process.argv[2];
    const target = process.argv[3];
    const extra = process.argv[4] || 'en';
    
    const chromePath = process.env.CHROME_BIN || '/usr/bin/google-chrome';
    let browser;
    try {
        browser = await puppeteer.launch({
            executablePath: chromePath,
            headless: 'new',
            args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu', '--disable-dev-shm-usage']
        });
        const page = await browser.newPage();
        await page.setUserAgent('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36');
        
        let url = target;
        if (action === 'search_wikipedia') {
            url = `https://${extra}.wikipedia.org/w/index.php?search=${encodeURIComponent(target)}`;
        }
        
        await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 30000 });
        const title = await page.title();
        const textContent = await page.evaluate(() => {
            const toRemove = document.querySelectorAll('script, style, nav, footer, iframe, .navbox, noscript');
            toRemove.forEach(el => el.remove());
            return document.body ? document.body.innerText.replace(/\s+/g, ' ').trim() : '';
        });
        
        console.log(JSON.stringify({
            status: 'ok',
            url: page.url(),
            title: title,
            content: textContent.substring(0, 4000)
        }));
    } catch (err) {
        console.log(JSON.stringify({
            status: 'error',
            error: err.message
        }));
    } finally {
        if (browser) {
            await browser.close();
        }
    }
})();
