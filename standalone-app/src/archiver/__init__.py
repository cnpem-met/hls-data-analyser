import asyncio
import aiohttp
from qasync import asyncSlot

from config import ARCHIVER_URL

@asyncSlot()
async def fetch(session, pv, time_from, time_to, is_optimized, mean_minutes):
    """ Fetch data from Archiver """
    if is_optimized:
        pv_query = f'mean_{int(60*mean_minutes)}({pv})'
    else:
        pv_query = pv
    query = {'pv': pv_query, 'from': time_from, 'to': time_to}
    async with session.get(ARCHIVER_URL, params=query) as response:
        response_as_json = await response.json()
        return response_as_json

@asyncSlot()
async def retrieve_data(pvs, time_from, time_to, isOptimized=False, mean_minutes=0):
    """ mid function that fetches data from multiple pvs """
    async with aiohttp.ClientSession() as session:
        data = await asyncio.gather(*[fetch(session, pv, time_from, time_to, isOptimized, mean_minutes) for pv in pvs])
        return data
