import type { CodeSnapshot, Run } from './types';

export function trainingEnvironment(run: Run, snapshot?: CodeSnapshot) {
  const recorded = snapshot || run.attempts?.at(-1)?.code_snapshot || run.code_expectation;
  const profile = recorded?.runtime_profile;
  return {
    id: run.profile_id,
    name: profile?.name || run.profile_id,
    codeDir: recorded?.code_dir || profile?.code_dir,
    python: profile?.python,
  };
}

export function TrainingEnvironmentDetails({ run }: { run: Run }) {
  const environment = trainingEnvironment(run);
  return (
    <>
      <div>
        <dt>运行环境</dt>
        <dd>{environment.name}</dd>
      </div>
      <div>
        <dt>环境标识</dt>
        <dd>
          <code>{environment.id}</code>
        </dd>
      </div>
      <div>
        <dt>代码目录</dt>
        <dd>{environment.codeDir ? <code>{environment.codeDir}</code> : '未记录'}</dd>
      </div>
      <div>
        <dt>Python 环境</dt>
        <dd>{environment.python ? <code>{environment.python}</code> : '未记录'}</dd>
      </div>
    </>
  );
}
