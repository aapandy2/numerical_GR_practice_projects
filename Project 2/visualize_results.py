import numpy as np
import pylab as pl
import subprocess

# Set plot parameters to make beautiful plots
pl.rcParams['figure.figsize']  = 16, 9
pl.rcParams['lines.linewidth'] = 1.5
pl.rcParams['font.family']     = 'serif'
pl.rcParams['font.weight']     = 'bold'
pl.rcParams['font.size']       = 15
pl.rcParams['font.sans-serif'] = 'serif'
pl.rcParams['text.usetex']     = True
pl.rcParams['axes.linewidth']  = 1.5
pl.rcParams['axes.titlesize']  = 'large'
pl.rcParams['axes.labelsize']  = 'large'

pl.rcParams['xtick.major.size'] = 8
pl.rcParams['xtick.minor.size'] = 4
pl.rcParams['xtick.major.pad']  = 8
pl.rcParams['xtick.minor.pad']  = 8
pl.rcParams['xtick.color']      = 'k'
pl.rcParams['xtick.labelsize']  = 'large'
pl.rcParams['xtick.direction']  = 'in'

pl.rcParams['ytick.major.size'] = 8
pl.rcParams['ytick.minor.size'] = 4
pl.rcParams['ytick.major.pad']  = 8
pl.rcParams['ytick.minor.pad']  = 8
pl.rcParams['ytick.color']      = 'k'
pl.rcParams['ytick.labelsize']  = 'large'
pl.rcParams['ytick.direction']  = 'in'

#LOAD DATAFILES
r_grid      = np.loadtxt('r_grid.txt')
phi         = np.loadtxt('phi.txt')
xi          = np.loadtxt('xi.txt')
Pi          = np.loadtxt('Pi.txt')
psi         = np.loadtxt('psi.txt')
beta        = np.loadtxt('beta.txt')
alpha       = np.loadtxt('alpha.txt')
mass_aspect = np.loadtxt('mass_aspect.txt')

step      = 1
timesteps = np.shape(phi)[0]
N         = np.shape(phi)[1]

def make_movie(plot_variables='matter'):

	#make temp folder to save frames which will be made into the movie
	command0 = subprocess.Popen('mkdir temp_folder/'.split(), stdout=subprocess.PIPE)
	command0.wait()
	
	#save frames to make movie
	for i in range(0, timesteps, step):
	    if(i % 50 == 0):
	        print 'saving frame ' + str(i) + ' out of ' + str(timesteps)
	    figure, ax = pl.subplots(nrows=1, ncols=3, sharex=True, sharey=False)
	
	    if(plot_variables == 'matter'):
		    ax[0].plot(r_grid, phi[i, :])
		    ax[0].set_title('$$\\phi$$')
		    ax[1].plot(r_grid, xi[i, :])
		    ax[1].set_title('$$\\xi$$')
		    ax[2].plot(r_grid, Pi[i, :])
		    ax[2].set_title('$$\\Pi$$')
	    elif(plot_variables == 'geometry'):
		    ax[0].plot(r_grid, psi[i, :])
		    ax[0].set_title('$$\\psi$$')
		    ax[1].plot(r_grid, beta[i, :])
		    ax[1].set_title('$$\\beta$$')
		    ax[2].plot(r_grid, alpha[i, :])
		    ax[2].set_title('$$\\alpha$$')
	    elif(plot_variables == 'mass_aspect'):
		    ax[0].plot(r_grid, mass_aspect[i, :])
		    ax[0].set_title('Mass Aspect')
		    ax[1].axhline(np.amax(mass_aspect[i, :]))
		    ax[1].set_title('max(Mass Aspect)')
		    ax[2].axhline(mass_aspect[i, 0])
		    ax[2].set_ylim(-0.1, 0.1)
		    ax[2].set_title('m(r = 0)')
	
	    #draw x label $r$
	    figure.add_subplot(111, frameon=False)
	    pl.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
	    pl.xlabel('$r$', fontsize='large')
	
	   #set y limits on plots
#	    ax[0].set_ylim(-0.2, 1.)
#	    ax[1].set_ylim(0., 1.5e7)
#	    ax[2].set_ylim(-10., 10.)
	
	   #save frames, close frames, clear memory
	    pl.tight_layout()
	    pl.savefig('temp_folder/%03d'%(i/step) + '.png')
	    pl.close()
	    pl.clf()

	#make movie
	command1 = subprocess.Popen( ('ffmpeg -y -i temp_folder/%03d.png ' + plot_variables + '.m4v').split(), stdout=subprocess.PIPE)
	command1.wait()
	command2 = subprocess.Popen('rm -r temp_folder/'.split(), stdout=subprocess.PIPE)
	command2.wait()

	return 0.

make_movie('matter')
make_movie('geometry')
make_movie('mass_aspect')
