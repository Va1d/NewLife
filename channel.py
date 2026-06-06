import yt_dlp

def download_channel():
    ydl_opts = {
        # 1. Performance & Identity
        'format': 'bestvideo[height<=1080]+bestaudio/best[height<=1080]',
        'socket_timeout': 122,
        'source_address': '0.0.0.0',
        'ignoreerrors': True,
        'retries': 10,         
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36',
        },

        # 2. Paths & Auth
        'cookiefile': '/home/bo/Py/cookies.txt', 
        'download_archive': '/home/bo/Py/finished_episodes.txt',
        'outtmpl': '/home/bo/Py/Scott//%(title)s/%(title)s.%(ext)s',
        
        # 3. Solver
        'remote_components': 'ejs:github', 
        'js_runtimes': {
            'node': {'path': '/home/bo/.nvm/versions/node/v24.13.1/bin/node'}
        },
        
        # 4. Formats & Subs
        'merge_output_format': 'mp4',
        'keepvideo': True,           
        'writesubtitles': True,             
        'writeautomaticsub': True,          
        'subtitleslangs': ['en.*'],   

        # 5. Processing Logic
        'postprocessors': [
            {
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',            
            },
            {
                'key': 'FFmpegSubtitlesConvertor',
                'format': 'srt',
            }
        ],
        
        'postprocessor_args': {
            # 'default' handles the initial merge (creating the Stereo MP4)
            'default': ['-ac', '2'],                
            # Matches the 'key' in postprocessors exactly (creating the Mono WAV)
            'FFmpegExtractAudio': ['-ac', '1', '-ar', '44100'], 
        },
    }    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download('https://www.youtube.com/@RealCoffeewithScottAdams')
if __name__ == "__main__":
    download_channel()