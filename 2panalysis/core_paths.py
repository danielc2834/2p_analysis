import os, glob
from pathlib import Path
from datetime import date
from typing import ClassVar, Set, Any

class dataset_layout:
    """all Paths, relative to dataset-folder, and information relevant for further steps"""

    _FILE_ATTRS: ClassVar[Set[str]] = {"errorlog", "analysis_log"}

    def __init__(self, folder: str, today: date):
        """Initializes all fixed paths
        
        Parameter
        ---------
        folder : str
            Path to dataset folder, output of UI
        today: date
            date in format of date library
        """
        homepath = Path(folder)
        self.today = today
        self.sort = homepath / '0_to_sort'
        self.raw = homepath / '1_raw_recordings'
        self.processed = homepath / '2_processed_recordings'
        self.data_pkl = homepath / '3_DATA'
        self.results = homepath / '4_results'
        self.name = os.path.basename(folder)
        self.folder = homepath
        self.stim = homepath / '0_to_sort' / 'stim'
        self.zstacks = homepath /'5_ZStacks'
        self.stimdata = homepath / '6_stim_files'
        self.errorlog = homepath / f'error_log_{today}.txt'
        self.analysis_log = homepath  / "4_results" / f"analysis_log_{today}.txt"
    
    def ensure_paths(self) -> None:
        """
        Create any missing directories or log files
        """
        for attr_name, value in vars(self).items():
            if not isinstance(value, Path):
                continue
            is_file = (attr_name in self._FILE_ATTRS or value.suffix)                        
            if is_file:
                # Ensure the parent directory exists, then create an empty file.
                value.parent.mkdir(parents=True, exist_ok=True)
                value.touch(exist_ok=True)
            else:
                # Regular directory – create it (including any missing parents).
                value.mkdir(parents=True, exist_ok=True)

    def log_error(self, type: str, message: str):
        """
        write error into type specfici error logbook

        Parameter
        ---------
        type: str
            type of error log book (txt file) to write into. One of ['analysis', 'preprocessing']
        message: str
            error message that will be written into logbook. Line break is already inclueded
        """
        if type == 'analysis':
            if os.path.exists(self.analysis_log)==False:
                with open(self.analysis_log, 'w', encoding="utf8") as f:
                    f.write(f"{message}\n")
            else:
                try:
                    with open(self.analysis_log, 'a', encoding="utf8") as f:
                        f.write(f"{message}\n")
                except Exception:
                    pass 
        elif type == 'preprocessing':
            if os.path.exists(self.errorlog)==False:
                with open(self.errorlog, 'w', encoding="utf8") as f:
                    f.write(f"{message}\n")
            else:
                try:
                    with open(self.errorlog, 'a', encoding="utf8") as f:
                        f.write(f"{message}\n")
                except Exception:
                    pass 
        else: 
            pass

    def store_globals(self, **kwargs: Any) -> None:
        """
        Store keyword arguments as lowercase attributes on the instance.

        Parameters
        ----------
        **kwargs : dict
            Arbitrary keyword arguments. Each key becomes a lowercase attribute name on the instance, and the value is assigned to it.

        Notes
        -----
        - Attribute names are converted to lowercase to enforce consistent naming.
        - If a key already exists as an attribute, it will be overwritten.
        - No type checking is performed — values are assigned as-is.
        """
        for key, value in kwargs.items():
            setattr(self, key.lower(), value)
    
    def fetch_files(self, path: str, filter : str=None, recursive: bool=True) -> list[str]:
        """fetches paths of files 
        
        Parameter
        --------
        path : str
            full path to the main folder
        filter : str, default = ``None``
            any form of filter to specify fetched files
        recursive : boolean, default = ``True``
            whether to fetch files recrsive
        
        Returns
        -------
        files : [str]
            list of paths to fetched files
        """
        if recursive and filter is not None:
            files = glob.glob(f'{path}/**/*{filter}', recursive = True) 
        elif recursive and filter is None:
            files = glob.glob(f'{path}/**/*', recursive = True) 
        elif not recursive and filter is None:
            files = glob.glob(f'{path}/*') 
        elif not recursive and filter is not None:
            files = glob.glob(f'{path}/*{filter}') 
        return files

    def fetch_all_conditions(self) -> list[str]:
        '''
        collects all condition str of dataset in list

        Returns
        --------
        list[str] : 
            contains all experimental conditions of dataset
        '''
        if os.path.exists(self.raw) and len(os.listdir(self.raw))>1:
            return [d for d in os.listdir(self.raw) if os.path.isdir(os.path.join(self.raw, d))]
        elif os.path.exists(self.processed) and len(os.listdir(self.processed)) > 1:
            return [d for d in os.listdir(self.processed) if os.path.isdir(os.path.join(self.processed, d)) and not d.startswith('processing_progress')]
        else:
            self.log_error('analysis', 'No Conditions found')
            return []