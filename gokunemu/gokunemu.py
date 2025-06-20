import sys
sys.path.append("../")
import numpy as np
from mfbox import gokunet_df_ratio
from scipy.interpolate import interp1d

class MatterPowerEmulator:
    """
    Emulator class to predict the matter power spectrum P(k, z) using two neural networks.
    Combines two models at different k-ranges using a redshift-dependent transition point.
    """

    def __init__(self, device: str = 'cpu'):
        """
        Initialize emulator by loading neural network models and preparing redshift bins and stitching indices.
        """
        # Redshift bin setup
        lna = np.linspace(0, -np.log(1 + 3), 30)
        zs_uniform = 1 / np.exp(lna) - 1
        zs_manual = np.array([0, 0.2, 0.5, 1, 2, 3])
        self.z_bins = np.unique(np.concatenate((zs_uniform, zs_manual)))

        # Load pretrained neural networks
        bounds_path = "input_limits-W.txt"
        self.emu1 = gokunet_df_ratio(
            path_LF="../models/L1A/best_model.pth",
            path_LHr="../models/L1HAr/best_model.pth",
            bounds_path=bounds_path,
            device=device
        )
        self.emu2 = gokunet_df_ratio(
            path_LF="../models/L2/best_model.pth",
            path_LHr="../models/L2Hr/best_model.pth",
            bounds_path=bounds_path,
            device=device
        )

        # Precomputed k-transition values for z_bins
        self.k_trans_zs = np.array([
            1.36449757, 1.44150977, 1.44150977, 1.29159973, 1.29159973,
            1.29159973, 1.29159973, 1.29159973, 1.15727961, 1.29159973,
            1.29159973, 1.29159973, 1.29159973, 0.92909262, 0.92909262,
            0.7060491 , 1.29159973, 1.29159973, 1.29159973, 1.29159973,
            1.22259643, 1.22259643, 1.22259643, 1.15727961, 1.15727961,
            1.09545232, 1.09545232, 0.87945615, 0.98153058, 1.03692813,
            1.15727961, 1.03692813, 1.09545232, 1.15727961
        ])

        # Sample dummy input to extract k1 and k2 structure
        dummy_params = np.zeros((1, 10))
        self.k1, _ = self.emu1.predict(dummy_params)
        self.k2, _ = self.emu2.predict(dummy_params)

        # Precompute kAB and stitching indices per z-bin
        self.kB_extra = self.k2[self.k2 > self.k1[-1]]
        self.kAB = np.concatenate((self.k1, self.kB_extra))
        self.stitch_slices = []
        for k_trans in self.k_trans_zs:
            idxA = self.k1 <= k_trans
            idxB = self.k2 > k_trans
            len_A = np.sum(idxA)
            len_B = np.sum(idxB)
            self.stitch_slices.append((len_A, len_B))

    def predict(
        self,
        cosmo_params: np.ndarray = None,
        Om: float = 0.3,
        Ob: float = 0.05,
        hubble: float = 0.7,
        As: float = 2.1e-9,
        ns: float = 0.96,
        w0: float = -1.0,
        wa: float = 0.0,
        mnu: float = 0.06,
        Neff: float = 3.044,
        alphas: float = 0.0,
        redshifts: np.ndarray = np.array([0.])
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predicts the matter power spectrum for given cosmological parameters and redshifts.
        Uses precomputed kAB and stitching indices to speed up repeated inference.

        Returns:
        - kAB: Combined k-array
        - Pk_obj: Array of P(k, z) predictions, shape (n_samples, len(redshifts), len(kAB))
        """
        if cosmo_params is None:
            cosmo_params = np.array([[Om, Ob, hubble, As, ns, w0, wa, mnu, Neff, alphas]])
        else:
            cosmo_params = np.atleast_2d(cosmo_params)

        n_samples = cosmo_params.shape[0]
        n_z_bins = len(self.z_bins)

        # Predict full redshift grid with both emulators
        _, Pk1 = self.emu1.predict(cosmo_params)
        _, Pk2 = self.emu2.predict(cosmo_params)

        # Allocate combined spectrum
        Pk = np.zeros((n_samples, n_z_bins, len(self.kAB)))

        # Efficient stitching using precomputed slices
        for i, (len_A, len_B) in enumerate(self.stitch_slices):
            Pk[:, i, :len_A] = Pk1[:, i, :len_A]
            Pk[:, i, len_A:] = Pk2[:, i, -len_B:]

        # Interpolate log10 P(k) in redshift space to target redshifts
        log_Pk = np.log10(Pk)
        log_Pk_interp = np.zeros((n_samples, len(redshifts), len(self.kAB)))
        for i in range(n_samples):
            f = interp1d(self.z_bins, log_Pk[i], kind='linear', axis=0, bounds_error=False, fill_value='extrapolate')
            log_Pk_interp[i] = f(redshifts)

        return self.kAB, 10 ** log_Pk_interp
    
