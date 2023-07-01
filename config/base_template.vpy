import vapoursynth as vs
core=vs.core
core.num_threads=32
core.max_cache_size=32768
src=core.lsmas.LWLibavSource('__SRCFILE__')
out=core.sub.TextFile(src, '__ASSFILE__', fontdir='__FONTDIR__')
out.set_output(0)
aud=core.bs.AudioSource('__SRCFILE__')
aud.set_output(1)
