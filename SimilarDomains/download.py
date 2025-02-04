import os
import click
import subprocess

from pathlib import Path


def download_curl(source: str, destination: str) -> None:
    subprocess.run(
        ['curl', '-L', '-k', source, '-o', destination],
        stdout=subprocess.DEVNULL
    )

    
def untar(path: str, destination: str = None):
    command = ['tar', '-xzvf', path]
    
    if destination is not None:
        command += ['-C', destination]
        if not os.path.exists(destination):
            os.makedirs(destination)
    subprocess.run(command, stdout=subprocess.DEVNULL)
    
    
def download_gdrive(file_id: str, destination: str) -> None:
    subprocess.run(['gdown', '--id', file_id, '-O', destination])
    
    
def unzip(path: str, res_path: str = None):
    command = ['unzip', path]
    
    if res_path is not None:
        command += ['-d', res_path]
    subprocess.run(command, stdout=subprocess.DEVNULL)


def bzip2(path: str):
    subprocess.run(['bzip2', '-d', path])
    
    
def rm_file(path: str):
    subprocess.run(['rm', path])
    

class Setup:    
    def __init__(self):
        self.root = Path(__file__).parent
        self.pretrained_root = Path(__file__).parent / 'pretrained'
        self.pretrained_root.mkdir(exist_ok=True)
        
    def _download(self, data):
        
        if data.get('root_located', False):
            root = self.root
        else:
            root = self.pretrained_root
        
        file_dest = str(root / data['name'])
        
        if 'link' in data:
            download_curl(data['link'], file_dest)
        elif 'id' in data:
            download_gdrive(data['id'], file_dest)
        
        if file_dest.endswith('bz2'):
            bzip2(file_dest)
            rm_file(file_dest)
        elif file_dest.endswith('tar.gz'):
            untar(file_dest, str(root / data['uncompressed_dir']))
            rm_file(file_dest)
        elif file_dest.endswith('.zip'):
            unzip(file_dest, str(root / data['uncompressed_dir']))
            rm_file(file_dest)
            
    def setup(self, values):
        for value in values:
            self._download(SOURCES[value])


SOURCES = {
    'sg2-ffhq': {
        'link': 'https://nxt.2a2i.org/index.php/s/kyR9byFznz5GBTd/download/stylegan2-ffhq-config-f.pt.zip',
        'name': 'stylegan2-ffhq-config-f.pt.zip',
        'uncompressed_dir': ''
    },
    'dlib': {
        'link': 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2',
        'name': 'shape_predictor_68_face_landmarks.dat.bz2'
    },
    'restyle_psp': {
        'id': '1nbxCIVw9H3YnQsoIPykNEFwWJnHVHlVd',
        'name': 'restyle_psp_ffhq_encode.pt'
    },
    'e4e': {
        'link': 'https://nxt.2a2i.org/index.php/s/ey49AsRwgyK77C9/download/e4e_ffhq_encode.pt.zip',
        'name': 'e4e_ffhq_encode.pt.zip',
        'uncompressed_dir': ''
    },
    'clip_means': {
        'link': 'https://nxt.2a2i.org/index.php/s/CbxaqSy6C7sFNW2/download/clip_means.zip',
        'name': 'clip_means.zip',
        'uncompressed_dir': '',
        'root_located': True,
    },
    'ckpt': {
        'link': 'https://nxt.2a2i.org/index.php/s/eDWLK8rDzSFoxeZ/download/checkpoints.tar.gz',
        'name': 'checkpoints.tar.gz',
        'uncompressed_dir': 'checkpoints_iccv'
    },
    'sg2_tuned': {
        'link': 'https://nxt.2a2i.org/index.php/s/JzwG7gFHaKrHwDt/download/StyleGAN2_ADA.zip',
        'name': 'StyleGAN2_ADA.zip',
        'uncompressed_dir': ''
    },
    'sg2': {
        'link': 'https://nxt.2a2i.org/index.php/s/2K3jbFD3Tg7QmHA/download/StyleGAN2.zip',
        'name': 'StyleGAN2.zip',
        'uncompressed_dir': ''
    },
    'image_domains': {
        'link': 'https://nxt.2a2i.org/index.php/s/ZTBnffeW5TfrJjy/download/image_domains.zip',
        'name': 'image_domains.zip',
        'uncompressed_dir': '',
        'root_located': True,
    }
}


@click.command()
@click.argument('value', default=None, nargs=-1)
def main(value):
    downloader = Setup()
    values = value if value else SOURCES.keys()
    downloader.setup(values)
    

if __name__ == '__main__':
    main()
