import type {
  AgentConfigRequest,
  RemoteDeployment,
  SshAuthenticationRequest,
  SshConnectionResult,
  SshInstallCheckRequest,
} from './types';

export const DEFAULT_AGENT_SERVICE = 'termitech-vla-agent';

export function agentServiceError(value: string): string | undefined {
  return /^[A-Za-z0-9_-]{1,100}$/.test(value.trim())
    ? undefined
    : 'Agent 服务名须为 1 至 100 位字母、数字、连字符或下划线，不包含 .service。';
}

export function sshAuthenticationRequest(
  host: string,
  port: string,
  username: string,
  password: string,
  serviceName: string,
): SshAuthenticationRequest {
  return {
    host: host.trim(),
    port: +port,
    username: username.trim(),
    password,
    service_name: serviceName.trim(),
  };
}

export function sshInstallCheckRequest(
  installDir: string,
  serviceName: string,
): SshInstallCheckRequest {
  return { install_dir: installDir.trim(), service_name: serviceName.trim() };
}

export function deploymentService(job: Pick<RemoteDeployment, 'service_name'>): string {
  return job.service_name || DEFAULT_AGENT_SERVICE;
}

/** A new service must not inherit another service's machine identity or paths. */
export function checkedProbeConfiguration(
  result: SshConnectionResult,
  requestedService: string,
  installDir: string,
  fallbackHost: string,
): AgentConfigRequest {
  const selectedService = requestedService.trim();
  if (agentServiceError(selectedService)) throw new Error(agentServiceError(selectedService));
  if ((result.service_name || DEFAULT_AGENT_SERVICE) !== selectedService) {
    throw new Error('安装检查返回的 Agent 服务名不匹配，请更新控制台并重新检查连接。');
  }
  const prefix = installDir.trim().replace(/\/+$/, '');
  const existing = result.existing_agent;
  return {
    name: existing?.name || result.hostname || fallbackHost.trim(),
    roots: existing?.roots.length ? [...existing.roots] : [`/${prefix.split('/')[1]}`],
    output_root: existing?.output_root || `${prefix}/outputs`,
    state_dir: existing?.state_dir || `${prefix}/state`,
    profiles: existing
      ? existing.profiles.map((profile) => ({ ...profile }))
      : [
          {
            id: 'openpi-native-1',
            name: 'OpenPI 训练环境 1',
            python: '',
            code_dir: '',
            adapter: 'openpi-sft',
            execution_mode: 'native',
          },
        ],
  };
}
