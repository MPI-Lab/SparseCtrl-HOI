import subprocess


def add_silence_to_audio_ffmpeg(audio_path, tmp_audio_path, silence_duration_s=0.5):
    """Prepend ``silence_duration_s`` seconds of silence to ``audio_path`` (16 kHz stereo)."""
    command = [
        'ffmpeg',
        '-i', audio_path,
        '-f', 'lavfi',
        '-t', str(silence_duration_s),
        '-i', 'anullsrc=r=16000:cl=stereo',
        '-filter_complex', '[1][0]concat=n=2:v=0:a=1[out]',
        '-map', '[out]',
        '-y', tmp_audio_path,
        '-loglevel', 'quiet'
    ]
    subprocess.run(command, check=True)
