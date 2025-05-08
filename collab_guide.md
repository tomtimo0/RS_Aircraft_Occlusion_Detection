<details><summary>Git、GitHub操作与团队协作指南 (点击展开/折叠)</summary>

### 一、常用 Git 操作详解

1.  **克隆仓库 (Cloning)**
    *   **目的**: 当团队其他成员需要获取项目代码时，他们需要克隆远程 GitHub 仓库到自己的本地电脑。
    *   **命令**: `git clone <repository_url>`
        *   例如: `git clone https://github.com/tomtimo0/RS_Aircraft_Occlusion_Detection.git`
    *   **说明**: 这会在当前目录下创建一个与远程仓库同名的文件夹，并包含所有项目文件和版本历史。

2.  **查看状态 (Checking Status)**
    *   **目的**: 了解当前工作区文件的状态（哪些文件被修改了、哪些文件已暂存等）。
    *   **命令**: `git status`
    *   **说明**: 这是最常用的命令之一，建议在执行 `add`、`commit` 前都运行一下。

3.  **创建与切换分支 (Branching)**
    *   **目的**: 为了并行开发不同功能或修复 bug，而不影响主分支 (`main`) 的稳定性。每个新功能、修复都应该在新的分支上进行。
    *   **创建新分支并切换过去**: `git checkout -b <branch_name>`
        *   例如: `git checkout -b feature/data-preprocessing` 或 `git checkout -b fix/readme-typo`
    *   **切换到已存在的分支**: `git checkout <branch_name>`
    *   **查看所有本地分支**: `git branch`
    *   **查看所有远程分支**: `git branch -r`
    *   **查看所有本地和远程分支**: `git branch -a`
    *   **删除本地分支 (谨慎操作)**: `git branch -d <branch_name>` (如果已合并) 或 `git branch -D <branch_name>` (强制删除)

4.  **暂存更改 (Staging Changes)**
    *   **目的**: 将工作目录中已修改或新增的文件添加到 Git 的暂存区，准备进行提交。
    *   **暂存特定文件**: `git add <file_name>`
        *   例如: `git add README.md`
    *   **暂存特定文件夹下的所有更改**: `git add <folder_name>/`
    *   **暂存当前目录下所有更改 (包括新增、修改、删除)**: `git add .` (常用) 或 `git add -A`

5.  **提交更改 (Committing Changes)**
    *   **目的**: 将暂存区的内容永久保存到本地仓库的版本历史中。
    *   **命令**: `git commit -m "Your descriptive commit message"`
        *   例如: `git commit -m "Feat: Implement Gaussian blur for occlusion simulation"`
    *   **说明**:
        *   提交信息 (`commit message`) 非常重要，应该清晰、简洁地描述本次提交做了什么。
        *   遵循一定的提交规范（如 Conventional Commits）有助于生成清晰的变更日志。例如：`feat:` (新功能), `fix:` (bug修复), `docs:` (文档修改), `style:` (代码格式), `refactor:` (代码重构), `test:` (测试相关), `chore:` (构建过程或辅助工具变动)。

6.  **推送更改到远程仓库 (Pushing Changes)**
    *   **目的**: 将本地仓库的提交同步到 GitHub 远程仓库，让团队成员可以看到你的更改。
    *   **命令**: `git push origin <branch_name>`
        *   例如: `git push origin feature/data-preprocessing`
    *   **说明**:
        *   `origin` 是你远程仓库的默认名称。
        *   第一次推送新创建的本地分支到远程时，可能需要使用 `git push -u origin <branch_name>`，`-u` 参数会将本地分支与远程分支关联起来，之后可以直接使用 `git push`。

7.  **拉取远程仓库的更改 (Pulling Changes)**
    *   **目的**: 获取远程仓库特定分支的最新更改并合并到你当前的本地分支。
    *   **命令**: `git pull origin <branch_name>`
        *   例如，如果你在本地的 `feature/data-preprocessing` 分支，想拉取远程 `main` 分支的更新并合并进来:
            1.  `git checkout main` (切换到本地main分支)
            2.  `git pull origin main` (更新本地main分支)
            3.  `git checkout feature/data-preprocessing` (切换回你的特性分支)
            4.  `git merge main` (将最新的main分支合并到你的特性分支)
        *   或者，更直接的方式（在你的特性分支上）: `git pull origin main` (这会尝试获取远程main并合并到你当前分支，但有时更推荐先更新本地main再合并)
    *   **`git pull` 实际上是 `git fetch` 和 `git merge FETCH_HEAD` 的组合。**
    *   **使用 rebase 拉取**: `git pull --rebase origin <branch_name>`
        *   这会先将你本地未推送的提交暂存，拉取远程更改，然后再将你的本地提交应用在远程更改的顶端。可以使提交历史更线性、更整洁，但如果多人协作同一分支，需要谨慎使用，并确保团队成员都理解其工作方式。

8.  **获取远程仓库的更新 (Fetching Changes)**
    *   **目的**: 从远程仓库下载最新的版本历史信息，但**不**自动合并到你的本地工作分支。
    *   **命令**: `git fetch origin`
    *   **说明**: `fetch` 后，你可以使用 `git log origin/main` 查看远程 `main` 分支的提交历史，然后决定是否以及如何合并（例如使用 `git merge origin/main` 或 `git rebase origin/main`）。这比 `git pull` 提供了更多控制权。

9.  **合并分支 (Merging Branches)**
    *   **目的**: 将一个分支的更改合并到另一个分支。
    *   **命令**:
        1.  首先切换到你想要并入更改的目标分支: `git checkout <target_branch>` (例如 `git checkout main`)
        2.  然后执行合并命令: `git merge <source_branch>` (例如 `git merge feature/data-preprocessing`)
    *   **解决合并冲突**: 如果两个分支修改了同一个文件的同一部分，Git 无法自动合并，就会产生合并冲突。你需要手动打开冲突文件，编辑解决冲突（Git 会用特殊标记符如 `<<<<<<<`, `=======`, `>>>>>>>` 标出冲突部分），然后 `git add <resolved_file>`，最后 `git commit` 完成合并。

10. **查看提交历史 (Viewing Log)**
    *   **目的**: 查看项目的提交记录。
    *   **命令**:
        *   `git log`: 显示详细历史。
        *   `git log --oneline`: 每条提交显示为一行。
        *   `git log --graph --oneline --decorate --all`: 以图形化方式显示所有分支的简洁历史。
        *   `git log <file_name>`: 查看特定文件的修改历史。

### 二、团队协作注意事项 (GitHub)

有效的团队协作是项目成功的关键。

1.  **统一的开发流程 (Git Workflow)**
    *   **推荐流程：Feature Branch Workflow + Pull Requests**
        1.  **`main` 分支保持稳定**: `main` 分支应该始终代表可部署/稳定的代码。不允许直接向 `main` 分支推送代码。
        2.  **创建特性分支**: 对于每一个新功能、bug修复或任务（对应 `README.md` 中的 To-Do List 条目），都从最新的 `main` 分支创建一个新的、描述性的分支。
            *   分支命名规范：`feature/task-description` (如 `feature/gaussian-occlusion`), `fix/bug-description` (如 `fix/dota-loader-error`), `docs/update-readme`, `chore/setup-ci`。
        3.  **本地开发和提交**: 在你的特性分支上进行开发，频繁地进行小的、原子性的提交。
        4.  **定期拉取 `main` 更新**: 定期将 `main` 分支的最新更改合并到你的特性分支，以尽早发现和解决冲突：
            ```bash
            git checkout main
            git pull origin main
            git checkout your-feature-branch
            git merge main
            # (解决冲突，然后提交)
            ```
        5.  **推送特性分支**: 将你的特性分支推送到 GitHub。
        6.  **创建 Pull Request (PR)**: 当特性开发完成并通过初步测试后，在 GitHub 上为你的特性分支创建一个 Pull Request，目标是合并到 `main` 分支。

2.  **Pull Requests (PRs / 合并请求)**
    *   **清晰的描述**: PR 描述应清楚说明：
        *   **做了什么？ (What)** 实现了什么功能或修复了什么bug。
        *   **为什么要做？ (Why)** 解决什么问题，对应哪个 Issue (如果使用)。
        *   **如何做的？ (How - 可选)** 简要说明实现思路或关键改动。
        *   **如何测试？ (可选)** 提供测试步骤或结果。
    *   **代码审查 (Code Review)**:
        *   至少邀请**一位**团队成员审查你的代码。5人小组可以互相审查。
        *   审查者关注：代码逻辑、是否符合需求、可读性、潜在bug、是否遵循编码规范、测试覆盖等。
        *   PR 作者应积极回应审查意见并进行修改。
    *   **自动化检查 (可选但推荐)**: 如果项目配置了CI/CD (持续集成/持续交付)，PR可以触发自动化测试、代码风格检查等。
    *   **合并PR**: 只有当PR通过审查（和自动化检查），并且解决了所有提出的问题后，才由指定人员（如组长或主要开发者）将其合并到 `main` 分支。GitHub 提供了合并选项（如 Merge commit, Squash and merge, Rebase and merge）。对于课程设计，普通的 "Merge commit" 通常足够。

3.  **使用 GitHub Issues**
    *   **任务跟踪**: 为 To-Do List 中的每个主要任务、发现的 bug、或提出的改进建议创建一个 Issue。
    *   **分配责任**: 将 Issue 分配给具体的团队成员。
    *   **标签 (Labels)**: 使用标签对 Issues 进行分类 (如 `bug`, `enhancement`, `documentation`, `data-preprocessing`, `model-training`, `high-priority`)。
    *   **里程碑 (Milestones - 可选)**: 可以将一组相关的 Issues 归到一个里程碑，对应项目的主要阶段（如"数据准备阶段完成"、"模型初版训练完成"）。
    *   **关联PR**: 在 PR 描述中通过 `#issue_number` (如 `Closes #12`) 将 PR 与相关的 Issue 关联起来，这样合并 PR 时可以自动关闭对应的 Issue。

4.  **沟通与代码风格**
    *   **清晰的 Commit Messages**: 如前所述，非常重要。
    *   **定期同步**: 定期开小组会议（例如每周一次），同步进度，讨论遇到的问题，规划下一步工作。
    *   **编码规范**:
        *   对于 Python，遵循 **PEP 8** 风格指南。
        *   可以使用代码格式化工具如 **Black** 自动统一代码风格。
        *   使用 Linter 工具如 **Flake8** 或 **Pylint** 检查代码风格和潜在错误。
        *   在项目开始前，团队可以一起商定一些代码编写约定。
    *   **文档**:
        *   `README.md` 是项目的门面，保持其更新，包括：项目简介、如何安装依赖、如何运行代码（训练、测试、演示）、To-Do List 进展。
        *   对复杂函数和模块编写清晰的文档字符串 (docstrings)，遵循 JSDoc 风格（对于 Python 则是 reStructuredText 或 Google/Numpy 风格的 docstrings）。你之前提到的JSDoc主要是JavaScript的，Python有其自身的docstring规范。

5.  **`.gitignore` 文件**
    *   确保 `.gitignore` 文件配置正确，以避免将不必要的文件（如IDE配置文件、虚拟环境文件夹 `venv/` 或 `.env`、Python 编译的 `.pyc` 文件、大型数据集文件、模型权重文件等）提交到仓库。
    *   对于大型文件（如数据集、预训练模型），考虑使用 Git LFS (Large File Storage) 或云存储服务，并在 `.gitignore` 中忽略这些文件本身，只提交 LFS 指针或下载脚本。

6.  **保护 `main` 分支 (可选但推荐)**
    *   在 GitHub 仓库的 Settings -> Branches 中，可以为 `main` 分支设置保护规则，例如：
        *   禁止直接推送 (Require pull request reviews before merging)。
        *   要求至少 N 个审查者批准。
        *   要求通过状态检查 (如 CI 测试通过)。

</details> 