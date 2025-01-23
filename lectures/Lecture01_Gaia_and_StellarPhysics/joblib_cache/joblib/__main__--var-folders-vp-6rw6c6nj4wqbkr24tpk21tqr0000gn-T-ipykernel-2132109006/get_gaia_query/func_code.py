# first line: 1
@memory.cache
def get_gaia_query(q):
    start = time.time()
    job = Gaia.launch_job(q)
    print(f"Total time: {time.time()-start:0.2f} sec")
    return job.get_results()
