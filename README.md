# pi2020_imagematch
Find matches in images. Image processing project

## Development setup for Windows

We need the following:
 - An IDE, we will work with Visual Studio Code
 - A python environment, we will use Anaconda (miniconda)
 - A secure way to connect to the versioning system (github)

### Security

We will create a pair of ssh keys that will secure the link with the versioning system. 
In windows 10 we have OpenSSH installed.

Open a console and type

    ssh

if it outputs some help, you have OpenSSH. If it errs, install OpenSSH client: 

https://docs.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse

Then, create a key with 4096 bytes (run in a terminal):

    ssh-keygen -b 4096

Note where it is saved, and note which one is the public key.

### Git and github.

#### github

First, github. Go to www.github.com, create an account if you don't have one.
Make sure that you create a nice user name and fill in actual details. It will probably stick for a while.

Go to your user icon -> Settings -> SSH and GPC keys -> New SSH key.

Copy/paste here the contents of id_rsa.pub (PUBLIC KEY) generated with ssh-keygen.

Add some description and click ok.

#### git

Now, install git:

https://git-scm.com/

Leave most of the settings to default. Make sure that OpenSSH is selected, instead of Putty.

Test the git installation. Open the git terminal, from start menu (git bash or git cmd) and enter:

git --version

Should output something meaningful.

#### GitExtensions

One can try to work with git from command line but it is not easy or safe for the beginners.

We will install Gitextensions: https://github.com/gitextensions/gitextensions/releases/latest

Start GitExtensions and go to Settings. Fill in your name and email. (they will appear in each commit)

Now, clone this repository:

In GitExtensions, go to Start -> Clone Repository and at repository name paste: git@github.com:cristi-zz/pi2020_imagematch.git

[The link can be found at the repository's web page, under "Clone or Download" button. Select SSH as method.]

Note the destination folder.

### Python and IDE

#### Python environment

Python relies heavily on paths. All the necessary files should be in specific locations. In order to 
manage the whole complexity we will use a distribution manager, Anaconda.

Go to https://docs.conda.io/en/latest/miniconda.html and download latest miniconda.

Open an anaconda CMD (from start menu). It should open a regular cmd prompt with the phrase "(base)" prepended.

Run:

    conda init powershell

Note the errors, (eg antivirus preventing powershell settings change)

Then, and create a new environment that will be called *pi*:

    conda create -y -c conda-forge --copy --name pi python=3.7 numpy opencv pytest

The operation should succeed without any errors. Close the shell.

#### Install Visual Studio Code

Finally, the IDE. Microsoft released some decent IDE for python. Install it from here:
https://code.visualstudio.com/

Go to Extensions and install Python extension (From Microsoft)

Now, open the cloned project. Open a python file (main.py) so the python extension will be triggered. 

Anaconda environment should be auto detected and *pi* should be offered as python interpreter.

Run main.py by selecting "Run python file in terminal..."

It should show a green image. Press a key to close the program.

[Download the test binaries and save them in binaries/ folder.]