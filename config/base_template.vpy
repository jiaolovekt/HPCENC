import vapoursynth as vs
core=vs.core
src=core.lsmas.LWLibavSource('__SRCFILE__')
out=core.sub.TextFile(src, '__ASSFILE__', fontdir='__FONTDIR__')
out.set_output()
