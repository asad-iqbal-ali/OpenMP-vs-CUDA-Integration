Asad Ali
MPCS 51087
HW5

Files:
omp
	simpson_omp.c: Simpson's Rule OpenMP Code
	mc_omp.c: Monte carlo OpenMP code
	class-func.c/.h: a few helper functions that I should've just included
in these files for every assignment...
	omp-test.sbatch: sbatch for running the tests I used
	makefile: make! Just remember to load the cuda module first. Target
"clean" removes all output files.

cuda
	simpson.cu: Simpson's Rule CUDA code
	mc.cu: Monte carlo OMP code
	class-func.cu/.h: same as above
	cuda-test.sbatch: same as above
	makefile: same as above

Use:
	OMP:
	./(executable) (input size) (# of threads)
	CUDA:
	./(executable) (input size)

	For Simpson, "input size" is the number of divisions across the domain
of the integral. For MC, it's the number of samples.

	All four files append to an output called (executable).dat, which
lists the calculated integral, number of threads (OMP) or number of threads
per block (CUDA), time to calculate, and input size.

Comments:
	MP versions were fairly straightfoward, but CUDA presented some
challenges. Larger inputs quickly become too big to fit into a single function
call (they hit the limit on too many blocks for a call), so I had to split it
up into separate calls depending on the input size. I'm not sure if it comes
out as exactly the right number of total threads, but as a result of having to
hack up the input so much (for that, and for determining the Threads Per
Block), the CUDA codes really only work with inputs that lead to a number of
total calculations that is divisible by the TPB size. As a result, the
Simpson's Rule algorithm is run with n as a multiple of 4 (since it's really
easy to find multiples of 32 that divide (4^5)), and the MC algorithm is run
with 192 * various powers of 10.
	I chose 192 as my block size because I was working with the M2090,
specifically, and on that card each multiprocessor can take a maximum of 8
blocks or 1536 threads, and 8*192 = 1536, so I figured this maximized
occupancy on each multiprocessor in terms of simultaneous threads. I didn't
really take into account memory use or registers, since I figured that those
were relatively low-impact for this particular calculation. There may have
been more room for optimization in terms of planning for warps to take on new
blocks that were as far away from each other as possible (I believe prof.
Siegel mentioned that was an advantageous thing to do?), but that got a bit
too complicated for me. I also know that having that "if threadIdx.x == 0"
part really slows down a warp, so I kind of balanced having more of those
peppered throughout the warps for the sake of maximum simultaneous occupancy.
	For OpenMP, it was easy enough to separate out the function being
tested and the domain, but on CUDA that stuff is pretty much hard-coded. There
is zero portability in terms of using that code for any other function or
domain.

Results:
	The monte carlo method was just straight up slow and dumb. It's
possible that I just didn't optimize it well, but I think having to get random
numbers so often really dragged it down, particularly in CUDA, where those
poor tiny cores can't really do that kind of big boy stuff. I had experimented
originally with allocating an array of random doubles beforehand and just
picking from those, but that method causes serious problems with memory once
the input size gets large. The error rate for monte carlo was
also higher than Simpson's rule on all counts, making it pretty much useless
for this purpose. Interestingly, it was consistently about 0.001 higher than
the actual value (taken from Wolfram Alpha) in the OpenMP case. I'm not sure
why that is.
	Simpson's rule seemed to run pretty much the same on CUDA as OpenMP in
terms of speed and accuracy. 
