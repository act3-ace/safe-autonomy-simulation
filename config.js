// renovate config
Object.assign(process.env, {
    GIT_AUTHOR_NAME: 'Renovate Bot',
    GIT_AUTHOR_EMAIL: 'buildbot@act3-ace.com',
    GIT_COMMITTER_NAME: 'Renovate Bot',
    GIT_COMMITTER_EMAIL: 'buildbot@act3-ace.com',
  });
  
module.exports = {
    endpoint: process.env.CI_API_V4_URL,
    hostRules: [
        {
        matchHost: 'https://reg.git.act3-ace.com/',
        username: process.env.CI_REGISTRY_USER,
        password: process.env.CI_REGISTRY_PASSWORD,
        },
    ],
    regexManagers: [
        {
          fileMatch: ["(^|/).act3-pt.yml$"],
          matchStrings: ["# renovate: depName=(?<depName>.*?)\\s+version: (?<currentValue>.*)"],
          datasourceTemplate: "gitlab-tags",
          registryUrlTemplate: "https://git.act3-ace.com/",
          ignorePaths: ["**/templates/**"]
        },
    ],
    platform: 'gitlab',
    username: 'renovate-bot',
    autodiscover: true,
};