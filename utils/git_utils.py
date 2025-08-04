
import subprocess
from datetime import datetime, timedelta,timezone
from collections import defaultdict




# -------------------------命令行形式--------------------------------------

def cat_file_blob(repository_path, sha):
    cmd = ' '.join(['git', '-C', repository_path, 'cat-file', 'blob', sha])
    cat_file_process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,text=True,errors='ignore')
    result = cat_file_process.stdout.read()
    return result

def checkout_commit(repository_path, commit_sha):
    try:
        cmd = ['git', '-C', repository_path, 'checkout', commit_sha]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        print(f"[Error] Failed to checkout commit: {commit_sha}")
        return False


def diff_tree(repository_path, sha):
    cmd = ' '.join(['git', '-C', repository_path, 'diff-tree','--no-commit-id','-r', sha, '-- \*.java'])
    ls_results = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,text=True).stdout.read().splitlines()
    changed_files = []
    for line in ls_results:
        res = line.split('\t')
        flag = res[0].split() # flag[4] - D,M,A
        filesha = flag[3]
        changed_files.append((filesha,res[1]))
    return changed_files
    
def ls_tree(repository_path, sha):
    cmd = ' '.join(['git', '-C', repository_path, 'ls-tree', '-r', sha])
    ls_results = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,text=True).stdout.read().splitlines()
    return ls_results

def list_notes(repository_path, refs='refs/notes/commits'):
    cmd = ' '.join(['git', '-C', repository_path, 'notes', '--ref', refs, 'list'])
    notes_lines = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,text=True).stdout.read().splitlines()
    return notes_lines

def get_buggy_files(repository_path,since_date,until_date):
    cmd = ' '.join(['git', '-C', repository_path, 'log', '--since="%s"'%(since_date.strftime('%Y-%m-%d %H:%M:%S')), '--until="%s"'%(until_date.strftime('%Y-%m-%d %H:%M:%S')),'--grep', 'fix',
        '--grep', 'bug','-i' ,'--pretty=format:%H,%cd'])
    commit_results = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,text=True).stdout.read().splitlines()
    # commit_lists = []
    buggy_file2datelist = defaultdict(list)
    for line in commit_results:
        res = line.split(',')
        commit_date = datetime.strptime(res[1], "%a %b %d %H:%M:%S %Y %z")
        commit_date = commit_date.astimezone(timezone.utc)
        commit_date = commit_date.replace(tzinfo=None)
        changed_files = diff_tree(repository_path,res[0])
        for file_sha,path in changed_files:
            buggy_file2datelist[path].append(commit_date)
        # commit_lists.append((res[0],commit_date))
    return buggy_file2datelist

def get_commits(repository_path,date,k):
    target_date = date
    start_date = target_date - timedelta(days=k)
    cmd = ' '.join(['git', '-C', repository_path, 'log', '--since="%s"'%(start_date.strftime('%Y-%m-%d %H:%M:%S')), '--until="%s"'%(target_date.strftime('%Y-%m-%d %H:%M:%S')),'--grep', 'fix',
        '--grep', 'bug','-i' ,'--pretty=format:%H,%cd'])
    commit_results = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,text=True).stdout.read().splitlines()
    commit_lists = []
    for line in commit_results:
        res = line.split(',')
        commit_date = datetime.strptime(res[1], "%a %b %d %H:%M:%S %Y %z")
        commit_date = commit_date.astimezone(timezone.utc)
        commit_date = commit_date.replace(tzinfo=None)
        commit_lists.append((res[0],commit_date))
    return commit_lists

def add_note(repository_path,filesha,note,refs='refs/notes/commits'):
    cmd = ' '.join(['git', '-C', repository_path, 'notes', '--ref', refs, 'add','-f', '-F', '-', filesha])
    git_process = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,text=True)
    git_process.stdin.write(note.encode())
    print(git_process.stdout.readline().decode())
    try:
        outs, errs = git_process.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        git_process.kill()
        outs, errs = git_process.communicate()
    if outs is not None:
        print(f"Outs: {outs.decode()}")
    if errs is not None:
        print(f"Errs: {errs.decode()}")

def extractShaAndPath(lines):
    fileshas = []
    paths = []
    for line in lines:
        parts = line.split('\t')
        paths.append(parts[1])
        sha = parts[0].split(' ')[2]
        fileshas.append(sha)
    return fileshas, paths


    
    
# --------------------------------repo形式------------------------------------------------------------------
def add_note_repo(repo,filesha,note,refs='refs/notes/commits'):
    repo.notes('--ref', refs, 'add', '-m', note, filesha)


if __name__ == '__main__':
    git_dir = '/deepo_data/baselines/Bench4BL/data/Commons/COMPRESS/gitrepo'
    blob_sha = '4ddc67e18bbf05c7b22f6921818f3f3588c30635'
    # results = ls_tree(git_dir,'rel/1.4')
    results = get_commits(git_dir,datetime(2017,5,27),15)
    # results = cat_file_blob(git_dir,blob_sha)
    print(results)
    # methods_list = diff_tree_all_methods('/deepo_data/datasets/14Ye/projects/birt', '78e2ceb624')
    # print('\n'.join(methods_list))
    # content = cat_file_blob('/deepo_data/datasets/14Ye/projects/eclipse.platform.swt',
    #                         'a7b62ae597')
    # methods = get_methods(f'a7b62ae597.java',content)
    # print(methods)
    # print(len(methods))
    # print(methods)
    # for method in methods:
    #     print(method)
