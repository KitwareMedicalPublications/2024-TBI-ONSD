# Cold storage for code that get's moved out of notebooks but not actively used.
#

class AnnotatedData:
    '''
    This was code to parse Viame-Web's polygon annotation and interpolate as they only returned key frames
    '''
    COLS=['AnnotationID', 'DataID', 'FrameID', 'BX1', 'BY1', 'BX2', 'BY2', 'Confidence', 'TargetLength', 'Type', 'Unknown1', 'PolyData']

    def __init__(self):
        pass
    
    
    def _interp_polydata(self, n, pts1, pts2):
        incr = (pts2 - pts1) / n
        return [pts1 + incr*i for i in range(n+1)]
        
    def _parse_polydata(self, txt):
        PREFIX = '(poly) '
        N1 = len(PREFIX)
        
        return np.asarray([float(x) for x in txt[txt.find(PREFIX)+N1:].split(' ')]).reshape((-1,2))
    
    def load(self, vidpath, annotpath):
        self.vid = skvideo.io.vread(vidpath)
        self.df = pd.read_csv(get_annotation(vidpath), names=self.COLS, comment='#')
        self.frms = np.arange(self.vid.shape[0])
        
        # not a VIAME constraint, but for our data there should not be overlapping 'eye' or 'nerve' tracks
        self.eye_polys = [None] * len(self.frms)
        self.nerve_polys = [None] * len(self.frms)
        poly1 = None
        poly2 = None
        sfrm = None
        efrm = None
        annot_id = None
        mytype = None
        
        for idx, row in self.df.iterrows():
            if row['AnnotationID'] != annot_id: # covers the init case where annot_id is None
                poly1 = self._parse_polydata(row['PolyData'])
                sfrm = row['FrameID']
                
                annot_id = row['AnnotationID']
                mytype = row['Type']
                if mytype == 'eye':
                    self.eye_polys[sfrm] = poly1
                elif mytype == 'nerve':
                    self.nerve_polys[sfrm] = poly1
                    
            elif not pd.isna(row['PolyData']):
                poly2 = self._parse_polydata(row['PolyData'])
                efrm = row['FrameID']
                interps = self._interp_polydata(efrm-sfrm+1, poly1, poly2)
                
                if mytype == 'eye':
                    self.eye_polys[sfrm+1:efrm+1] = interps[1:]
                elif mytype == 'nerve':
                    self.nerve_polys[sfrm+1:efrm+1] = interps[1:]
                
                sfrm = efrm
                poly1 = poly2