
## Install

```
pip install playwright
playwright install chromium
```

## Set up

To run this script you need get your JWT token from 4gk.ru. To do that
1. login int the site, goto to package page. 
2. open dev tools (F12 in windows), network tab, find Authorization JWT token (note, it expires in ~20-30 min)
3. Start downloading the package file,
4. copy the token from the network tab.

provide your JWT tokein in .env file (`GQ_JWT=your_token`)

## Download docx

run:
`python scrape_by_api.py --start_id=4000 --end_id=4002 [--num_packs=2]`

## Parse docs

For that the script `parse_docx.py` is used.