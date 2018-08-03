import numpy as np
import pylab as pl
import subprocess
import sys

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


print '-----loading datafiles-----'

#LOAD DATAFILES
r_grid      = np.loadtxt('r_grid.txt')
phi         = np.loadtxt('phi.txt')
xi          = np.loadtxt('xi.txt')
Pi          = np.loadtxt('Pi.txt')
psi         = np.loadtxt('psi.txt')
beta        = np.loadtxt('beta.txt')
alpha       = np.loadtxt('alpha.txt')
mass_aspect = np.loadtxt('mass_aspect.txt')

#xi_residual     = np.loadtxt('xi_residual.txt')
#phi_residual    = np.loadtxt('phi_residual.txt')
#Pi_residual     = np.loadtxt('Pi_residual.txt')
#psi_residual    = np.loadtxt('psi_residual.txt')
#psi_ev_residual = np.loadtxt('psi_ev_residual.txt')
#beta_residual   = np.loadtxt('beta_residual.txt')
#alpha_residual  = np.loadtxt('alpha_residual.txt')
print '-----done loading datafiles-----'

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
		sys.stdout.flush() #testing if this flushes output to SLURM .out file

	    figure, ax = pl.subplots(nrows=1, ncols=3, sharex=True, sharey=False)
	
	    if(plot_variables == 'matter'):
		    ax[0].plot(r_grid, phi[i, :])
		    ax[0].set_title('$$\\phi$$')
		    ax[0].set_ylim(np.amin(phi), np.amax(phi))
		    ax[1].plot(r_grid, xi[i, :])
		    ax[1].set_title('$$\\xi$$')
		    ax[1].set_ylim(np.amin(xi), np.amax(xi))
		    ax[2].plot(r_grid, Pi[i, :])
		    ax[2].set_title('$$\\Pi$$')
                    ax[2].set_ylim(np.amin(Pi), np.amax(Pi))
	    elif(plot_variables == 'geometry'):
		    ax[0].plot(r_grid, psi[i, :])
		    ax[0].set_title('$$\\psi$$')
                    ax[0].set_ylim(np.amin(psi), np.amax(psi))
		    ax[1].plot(r_grid, beta[i, :])
		    ax[1].set_title('$$\\beta$$')
                    ax[1].set_ylim(np.amin(beta), np.amax(beta))
		    ax[2].plot(r_grid, alpha[i, :])
		    ax[2].set_title('$$\\alpha$$')
                    ax[2].set_ylim(np.amin(alpha), np.amax(alpha))
	    elif(plot_variables == 'mass_aspect'):
		    ax[0].plot(r_grid, mass_aspect[i, :])
		    ax[0].set_title('Mass Aspect')
                    ax[0].set_ylim(np.amin(mass_aspect), 1.01*np.amax(mass_aspect))
		    ax[1].axhline(np.amax(mass_aspect[i, :]))
		    ax[1].set_title('max(Mass Aspect)')
                    ax[1].set_ylim(-0.01*np.amax(mass_aspect), 1.01*np.amax(mass_aspect))
		    ax[2].axhline(mass_aspect[i, 0])
		    ax[2].set_title('m(r = 0)')
                    ax[2].set_ylim(-0.1*np.amax(mass_aspect), 0.1*np.amax(mass_aspect))
	    elif(plot_variables == 'matter_residuals'):
                    ax[0].plot(r_grid, phi_residual[i, :])
                    ax[0].set_title('$$\\phi~residual$$')
#                    ax[0].set_ylim(np.amin(phi_residual), np.amax(phi_residual))
                    ax[1].plot(r_grid, xi_residual[i, :])
                    ax[1].set_title('$$\\xi~residual$$')
#                    ax[1].set_ylim(-10.*np.mean(xi_residual), 10.*np.mean(xi_residual))
#                    ax[1].set_ylim(np.amin(xi_residual), np.amax(xi_residual))
                    ax[2].plot(r_grid, Pi_residual[i, :])
                    ax[2].set_title('$$\\Pi~residual$$')
#                    ax[2].set_ylim(-10.*np.mean(Pi_residual), 10.*np.mean(Pi_residual))
#                    ax[2].set_ylim(np.amin(Pi_residual), np.amax(Pi_residual))
	    elif(plot_variables == 'geometry_residuals'):
                    ax[0].plot(r_grid, psi_residual[i, :])
                    ax[0].set_title('$$\\psi~residual$$')
#                    ax[0].set_ylim(np.amin(psi_residual), np.amax(psi_residual))
#                    ax[0].plot(r_grid, psi_ev_residual[i, :])
#                    ax[0].set_title('$$\\psi~evolution~eqn~residual$$')
#                    ax[0].set_ylim(np.amin(psi_ev_residual), np.amax(psi_ev_residual))
                    ax[1].plot(r_grid, beta_residual[i, :])
                    ax[1].set_title('$$\\beta~residual$$')
#                    ax[1].set_ylim(np.amin(beta_residual), np.amax(beta_residual))
                    ax[2].plot(r_grid, alpha_residual[i, :])
                    ax[2].set_title('$$\\alpha~residual$$')
#                    ax[2].set_ylim(np.amin(alpha_residual), np.amax(alpha_residual))

	
	    #draw x label $r$
	    figure.add_subplot(111, frameon=False)
	    pl.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
	    pl.xlabel('$r$', fontsize='large')
	
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

print '-----make matter movie-----'
make_movie('matter')
#print '-----make matter residuals movie-----'
#make_movie('matter_residuals')
print '-----make geometry movie-----'
make_movie('geometry')
#print '-----make geometry residuals movie-----'
#make_movie('geometry_residuals')
print '-----make mass aspect movie-----'
make_movie('mass_aspect')
print '-----done-----'
